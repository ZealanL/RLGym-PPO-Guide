# Introduction

## Installing RLGym-PPO and rlgym-sim:
Here are the steps to install everything needed:
*Skip a step if you already have the thing!*
1. Install [Python](https://www.python.org/downloads/) (make sure to add it to your PATH/environment variables)
2. Install [Git](https://git-scm.com/downloads) (you can just click through the install with all of the default settings)
3. Install the `RocketSim` package with `pip install rocketsim`
4. Install the `rlgym_sim` package with `pip install git+https://github.com/AechPro/rocket-league-gym-sim@main`
5. [Download the asset dumper](https://github.com/ZealanL/RLArenaCollisionDumper/releases/tag/v1.0.0) and [follow its usage instructions](https://github.com/ZealanL/RLArenaCollisionDumper/blob/main/README.md) to make the `collision_meshes` folder (we will move this later)
6. If you have an NVIDIA GPU, install [CUDA v11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
7. Install PyTorch from [its website](https://pytorch.org/get-started/locally/) (if you installed CUDA, select the CUDA version, otherwise select CPU)
8. Make a folder for your bot
9. Install RLGym-PPO with `pip install git+https://github.com/AechPro/rlgym-ppo`
10. Steal [example.py](https://github.com/AechPro/rlgym-ppo/blob/main/example.py) from RLGym-PPO and add it to your bot folder
11. Move `collision_meshes` to your bot folder

### Wait, where is Rocket League involved?
RLGym-PPO uses rlgym-sim, which is a version of RLGym that runs on a simulated version of Rocket League, without actually running the game itself. This means that you can use RLGym-PPO on non-windows platforms, without having Rocket League, and can also collect data from the game much faster.

## Actually running your bot
Once you have installed RLGym-PPO, you can run your bot by running `example.py` (you should do this through a command prompt, instead of double-clicking).

This will start training the bot, and will report its results to *wandb*, a data platform that you can use to see graphs of various info as your bot trains. It will also print out a big list of stuff into the console, which I will elaborate on more in the next section.

## The basics of the training loop
Training is a process of:
 - **Collection**: The bot collects data from the environment (i.e. the bot plays the game at super-speed). Each data point during gameplay is called a **step**.
 - **Learning**: The learning algorithm uses those collected steps to update the brain of the bot.

Every time this cycle of learning happens it is called an **iteration**. After each iteration, the bot will print out a report, which will look something like this:

```
--------BEGIN ITERATION REPORT--------
Policy Reward: 0.03805
Policy Entropy: 0.80701
Value Function Loss: 0.01235

Mean KL Divergence: 0.00279
SB3 Clip Fraction: 0.02373
Policy Update Magnitude: 0.07114
Value Function Update Magnitude: 0.13493

Collected Steps per Second: 8,581.13182
Overall Steps per Second: 5,977.26097

Timestep Collection Time: 4.31789
Timestep Consumption Time: 2.25241
PPO Batch Consumption Time: 0.11061
Total Iteration Time: 5.57030

Cumulative Model Updates: 28
Cumulative Timesteps: 550,208

Timesteps Collected: 50,006
--------END ITERATION REPORT--------
```

Some of these terms are complicated and require lots of learning about ML to understand, so I'll just cover the simpler ones.

`Policy Reward`: This is how much reward the bot has collected, on average, in each **episode**.
An **episode** is a small piece of gameplay that ends once a certain condition is met, such as a goal being scored.

`Collected steps per second`: This is how many **steps** of gameplay are being collected every second. The better computer you have, the higher this number will be. 

A **step** is a tiny portion of time in the game. Rocket League physics runs at 120 FPS, and each of these physics frames are called a tick. A step is a handful of ticks, 8 by default. Since `120/8 = 15`, this means your bot is running at 15 updates a second. 

We'll talk more about this later.

`Overall steps per second`: Collected steps/sec is just for the collection phase, but what about the learning phase? Overall steps/sec includes the time it takes to learn, after collecting all the data. If you want to know how quickly your bot is running in general, this is the number to look at.

`Timestep Collection Time`: How long it took to collect all of the steps.

`Timestep Consumption Time`: How long it took to "consume" (organize and learn from) all of those steps.

`Cumulative Timesteps`: This is the total number of steps collected during your bot's lifetime. If you plan on making a good bot, expect to see numbers in the tens of billions someday!

*Fun fact: If your bot steps 15 times a second, 1 billion steps is ~18,500 hours of Rocket League. Bots do not go outside.*

## Ok, but how does it learn to play RL?

When your bot first starts learning, it has absolutely no idea what is going on. Its brain starts off completely random, so it just mashes random inputs.

In order to get our bot to actually learn something, it needs to have rewards. Rewards are things that happen in the game that you want the bot to do more or less often. The learning algorithm is designed to try to maximize how much reward the bot is getting, while also exploring new ways to get even more rewards.

Rewards can be both positive and negative, and are ultimately just numbers assigned to specific steps. Positive rewards encourage, negative rewards punish.

By default, your bot has 3 rewards:
 - `VelocityPlayerToBallReward()`: Positive reward for moving towards the ball, negative reward for moving away
 - `VelocityBallToGoalReward()`: Positive for having the ball move towards the opponent's goal, negative reward for having the ball move towards your own goal
 - `EventReward()`: Positive reward for scoring, negative reward for conceding (getting scored on), and a smaller reward for getting a demo

*You can find these rewards in `example.py`.*

As your bot continuous to mash random inputs, it will accidentally trigger a reward or punishment. Then, during the learning phase, the learning algorithm will try to adjust the bot's brain such that it is more/less likely to do the things that were rewarded/punished.

### Wait, why don't we just have a goal reward?
If you think about it, the only thing that ultimately matters for winning a game of Rocket League is scoring and not getting scored on, so why do we need other rewards?

Well, the answer is that bots are not very smart compared to humans. They can't plan out what they want to do, nor do they know what a car is, nor have they ever heard of balls or goals before. They need a lot more specific encouragement to learn how to move towards the ball, hit the ball, collect boost, and so on. 

A lot of the difficulty of making a bot is creating rewards that encourage the bot to do what you want, without limiting its ability to explore other options.

Also, for future reference, when I say **resetting the bot**, I mean resetting all learning back to nothing. Resetting the bot is a good choice for a number of reasons, and usually occurs when the bot is not improving, or a significant change needs to be made that would break the current bot.

## Diving into actually modifying our bot

So, now that you know some of the fundamental ideas, lets start actually messing with stuff.

If you haven't already, open `example.py` in the Python editor of your choice.

### Action parser

The first thing I recommend changing is the action parser.
This is set at the line:

```python
action_parser = ContinuousAction()
```

**Continuous** **actions** means the bot can use any partial input, which allows for more precise input. However, this is more difficult to use, and I do not recommend it as your first action parser.

An **action** is the combination of controller inputs the bot presses (throttle, steer, jump, boost, etc.), and an action parser converts the outputs of the bot's brain into these controller inputs.

Most bots use a **discrete action** parser, which separates every useful permutation of inputs into their own box, and the bot can control the car by picking a specific box of inputs.

Now before you go ahead and swap out `ContinuousAction` with `DiscreteAction`, beware:
`DiscreteAction` is actually `MultiDiscrete`, which is not what I described.
The fully-discrete[*] action parser is called `LookupAction`, and is not included by the library by default.

You can find it here: https://github.com/RLGym/rlgym-tools/blob/main/rlgym_tools/extra_action_parsers/lookup_act.py

Since action parsers define how your bot controls the car, changing it usually means resetting the bot.


### Rewards and weights

Down below the list of rewards is a list of numbers.
These numbers are the weights of each reward, which is how intensely they will influence the bot. `VelocityPlayerToBallReward()` is the lowest, `VelocityBallToGoalReward()` is 10x more influential, and `EventReward()` is 100x more influential than that.

**Event-type rewards** are rewards that activate once when a specific thing happens. They are usually important game events, like hitting the ball, shooting, scoring, etc, and so on.

**Continuous-type rewards** are rewards that are active *while* something is happening, and thus can run for many steps in a row. Since they happen so often, they are inherently stronger than event rewards, and usually should have far less weight.

`VelocityPlayerToBallReward` and `VelocityBallToGoalReward` are continuous rewards, whereas `EventReward` is.. well.. yeah.

Inside the constructor to `EventReward()` are sub-weights for different events. Each event will be multiplied by its sub-weight, then multiplied by the reward's weight after. Scoring is `1 * 10 = 10` reward total, whereas demos are only a tenth the reward of scaling, `0.1 * 10 = 1` reward total.

All rewards are eventually normalized in the learning algorithm (unless you specifically turn that off, which you probably shouldn't). This means that what actually matters is how rewards are weighted *in relation* to other rewards.

I recommend that you:
- Increase `VelocityPlayerToBallReward` a bit (it's very important in the early stages)
- Add `FaceBallReward` with a small weight (this will reward your bot for facing the ball, which is very helpful in the early stages of learning)

### Obs builder

**Obs** is short for observation, and it is how your bot perceives the game. An **obs builder** converts the current **state** of the game into inputs to your bot's brain.

The default obs builder is a decent starting point. I've found you can get moderately better results if you add car-relative positions and velocities, but that's a bit more advanced.

Making obs builders is quite tedious, and also very very high-risk. If you slightly mess something up, it could make your bot unable to play the game, or even worse, play the game but poorly (which is worse, because it is harder to tell that something is wrong).

Since the obs defines the input to your bot's brain, changing obs usually means resetting the bot.

### Number of players

Two variables control the amount of players in each game.

```python
spawn_opponents = True
team_size = 1
```

`spawn_opponents` means that each game will have an orange player for every blue player.
You probably want that, unless you want to train something very specific that doesn't involve other players.

`team_size` is the number of players on each team. I recommend starting with the default of `1`, as making bots that can play with teammates has added challenges. You can also have a bot that plays all modes (1v1, 2v2, and 3v3), but that is more advanced. 

### Terminal conditions
```py
terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]
```

These are a set of conditions that define when episodes end. If any of these conditions trigger, the episode is over. Basically all bots have a `GoalScoredCondition` (because otherwise the ball is going to stay inside the goal... which is weird), as well as a `NoTouchTimeoutCondition`.

The `NoTouchTimeoutCondition` ends the episode if no player has touched the ball for a certain amount of time (10 seconds by default). This is helpful, especially in the early stages, and prevents you from wasting time collecting a ton of data on two motionless bots who aren't doing anything or are stuck upside-down.

### State setters

Once the game is terminal, it needs to be reset. By default, it will be reset to kickoff.

However, for beginner bots, kickoff is usually not the best state setter. 

I recommend using the `RandomState` state setter, especially in the early stages of training. 
Its constructor takes 3 arguments: `ball_rand_speed`, `cars_rand_speed`, and `cars_on_ground`. 
I recommend you use `(True, True, False)`, as this will make the cars and ball start at a random location with random velocities. 
The cars will also spawn airborne half of the time, meaning they will quickly learn how to somewhat orient themselves in the air, too.

The state setter is an argument of `rlgym_sim.make()`, within your `build_rocketsim_env()`.

_____
[Back to Table of Contents](README.md)
