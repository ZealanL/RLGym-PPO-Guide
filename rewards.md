# Rewards

This section covers what rewards do, and how to create your own rewards to get the behavior you want.

## What rewards do

For every step of gameplay, there's a corresponding reward value: positive reward values are good, and negative reward values are bad.

The learning algorithm (PPO) is always trying to maximize both current and future rewards.

So, if you want your bot to do more of something, you can add a positive reward for doing that thing.

If you want your bot to do less of something, you can add a negative reward (called a penalty) for doing that thing.

## Reward functions

Reward functions are responsible for applying reward to a given state. They are run for each player, every step.
This is where all of your reward logic will take place.

NOTE: The reward function code that follows will assume you already have these imports:
```py
import numpy as np # Import numpy, the python math library
from rlgym_sim.utils import RewardFunction # Import the base RewardFunction class
from rlgym_sim.utils.gamestates import GameState, PlayerData # Import game state stuff
```

## A simple in-air reward

Here, we will look at an example reward function that rewards the player for being in the air.

```py
class InAirReward(RewardFunction): # We extend the class "RewardFunction"
    # Empty default constructor (required)
    def __init__(self):
        super().__init__()

    # Called when the game resets (i.e. after a goal is scored)
    def reset(self, initial_state: GameState):
        pass # Don't do anything when the game resets

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        
        # "player" is the current player we are getting the reward of
        # "state" is the current state of the game (ball, all players, etc.)
        # "previous_action" is the previous inputs of the player (throttle, steer, jump, boost, etc.) as an array
        
        if not player.on_ground:
            # We are in the air! Return full reward
            return 1
        else:
            # We are on ground, don't give any reward
            return 0
```

The actual reward logic here for the in-air reward is pretty simple:
```py
if not player.on_ground:
    # We are in the air! Return full reward
    return 1
else:
    # We are on ground, don't give any reward
    return 0
```

Here I use `on_ground`, a field of a player (players are `PlayerData`).
You can browse the other player fields in `PlayerData`'s source code, here: https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/gamestates/player_data.py

Note that all reward functions need to have:
1. A constructor
2. A function called when the game resets (after a terminal condition)
3. A function that returns the reward for a given player

There are other functions you can also implement, like `pre_step()`, but these are the fundamental ones.

Your reward functions should always output in a range up to 1, such as `[-1, 1]` or `[0, 1]`. 

## A speed-toward-ball reward

Now we will move on to a more advanced reward, involving the ball.

We want our player to hit the ball, so we will reward it for having speed in the direction of the ball.

```py
# Import CAR_MAX_SPEED from common game values
from rlgym_sim.utils.common_values import CAR_MAX_SPEED

class SpeedTowardBallReward(RewardFunction):
    # Default constructor
    def __init__(self):
        super().__init__()

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        pass

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # Velocity of our player
        player_vel = player.car_data.linear_velocity
        
        # Difference in position between our player and the ball
        # When getting the change needed to reach B from A, we can use the formula: (B - A)
        pos_diff = (state.ball.position - player.car_data.position)
        
        # Determine the distance to the ball
        # The distance is just the length of pos_diff
        dist_to_ball = np.linalg.norm(pos_diff)
        
        # We will now normalize our pos_diff vector, so that it has a length/magnitude of 1
        # This will give us the direction to the ball, instead of the difference in position
        # Normalizing a vector can be done by dividing the vector by its length
        dir_to_ball = pos_diff / dist_to_ball

        # Use a dot product to determine how much of our velocity is in this direction
        # Note that this will go negative when we are going away from the ball
        speed_toward_ball = np.dot(player_vel, dir_to_ball)
        
        if speed_toward_ball > 0:
            # We are moving toward the ball at a speed of "speed_toward_ball"
            # The maximum speed we can move toward the ball is the maximum car speed
            # We want to return a reward from 0 to 1, so we need to divide our "speed_toward_ball" by the max player speed
            reward = speed_toward_ball / CAR_MAX_SPEED
            return reward
        else:
            # We are not moving toward the ball
            # Many good behaviors require moving away from the ball, so I highly recommend you don't punish moving away
            # We'll just not give any reward
            return 0
```

*NOTE: From my testing, the `SpeedTowardBallReward` above is better than the default `VelocityPlayerToBallReward`, because it does not go negative.*

## Using multiple rewards

The `rlgym_sim` environment requires a single reward function. To allow multiple rewards in a single function, there exists `CombinedReward`. 
This reward function runs a set of rewards, multiplies each reward by a corresponding weight, and sums them together as the total reward.

You can see `CombinedReward` is already being used in `example.py` by default:
```py
rewards_to_combine = (VelocityPlayerToBallReward(),
                        VelocityBallToGoalReward(),
                        EventReward(team_goal=1, concede=-1, demo=0.1))
reward_weights = (0.01, 0.1, 10.0)

reward_fn = CombinedReward(reward_functions=rewards_to_combine, 
                            reward_weights=reward_weights)
```

Each reward in the `rewards_to_combine` has a corresponding weight in the `reward_weights` tuple. A good reason for each reward function to output from -1 to 1 is so that its maximum absolute output is determined by its weight.

To demonstrate how to add reward functions and weights, I'll add our new `InAirReward`, and replace `VelocityPlayerToBallReward` with our new `SpeedTowardBallReward` like so:

```py
rewards_to_combine = ( # I like to break open the parentheses like this
                        InAirReward(),
                        SpeedTowardBallReward(),
                        VelocityBallToGoalReward(),
                        EventReward(team_goal=1, concede=-1, demo=0.1)
                    )
reward_weights = (0.002, 0.01, 0.1, 10.0)

reward_fn = CombinedReward(reward_functions=rewards_to_combine, reward_weights=reward_weights)
```

Personally, I find having to keep track of the weight list separately very annoying.
Thankfully, `CombinedReward` has a static function called `from_zipped()`, which takes in pairs of reward functions and weights in one big list. This allows you to do the following:

```py

reward_fn = CombinedReward.from_zipped(
    # Format is (func, weight)
    (InAirReward(), 0.002),
    (SpeedTowardBallReward(), 0.01),
    (VelocityBallToGoalReward(), 0.1),
    (EventReward(team_goal=1, concede=-1, demo=0.1), 10.0)
)
```

## Zero-sum rewards

> :warning: *This section is a W.I.P. and needs more review!* :warning:

If you are rewarding a player for doing something is good, it only makes sense to equally punish the opponent.

Every good bot I have seen has used zero-sum rewards, either partially or completely.

A zero-sum reward can be implemented with the following logic: `player_reward = self_reward - opponent_reward`

My commented implementation of a zero-sum reward wrapper can be found here: [zero_sum_reward.py](code/zero_sum_reward.py)

### To zero-sum, or not to zero-sum

Some bot creators make every reward zero-sum, but I believe this is overkill, and from my testing, making certain rewards zero-sum slows down learning.

Based on this testing, my philosophy is the following: **A reward should only be zero-sum if it is beneficial for the opponent to prevent it.**

Examples of things that the opponent should be trying to prevent:
- Bumps/demos
- Flip resets
- Strong powershots
- Collecting boost
- Having speed

Examples of rewards that the opponent shouldn't worry about preventing as much:
- Speed flips
- Air roll in air
- Air reward/ground penalty

Most rewards tend to benefit from being zero-sum, but some rewards, such as those for gameplay/movement tuning, shouldn't be.

If a reward is zero-sum that shouldn't be, it will add noise to the overall reward of every player. 
This noise makes other reward signals weaker and will slow down learning.

I also recommend not using zero-sum rewards on specific behaviors like flip resets until your bot is skilled enough to start defending against them.

### Team rewards and "team spirit"

Another thing zero-sum rewards do is distribute rewards between teammates.

If you are training a bot in a team mode, and not using zero-sum rewards, you may notice the bot fighting over the ball with its teammates.
This obviously is not good teamplay!

The solution is to share reward among teammates.

Team spirit systems do this through a setting called `team_spirit`, which ranges from 0 to 1.

At `team_spirit = 0`, no reward is shared between teammates.
At `team_spirit = 1`, all reward is shared between teammates, with each player's reward being the average of their teams.

When training a bot, you generally want to start `team_spirit` very low, and slowly increase it as the bot improves, until it reaches `1`.

Having high team spirit too early in training will greatly slow down learning. 
This is because early learning mostly focuses on individual behaviors, not team ones.

Adding team spirit to zero-sum reward logic gives us this:
```py
avg_team_reward = ... # Average reward of everyone on our team (including us)

avg_opp_reward = ... # Average reward of all opponents

# As team spirit increases, our own reward will be replaced by our team's average reward
# After that, we then just subtract the average opponent reward
player_reward = (self_reward * (1 - team_spirit)) + (avg_team_reward * team_spirit) - average_opp_reward
```
_____
[Back to Table of Contents](README.md)