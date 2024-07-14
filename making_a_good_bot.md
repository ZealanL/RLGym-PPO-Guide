# Making a Good Bot

> :warning: *This section is a W.I.P. and needs more review!* :warning:

While most of this guide explains how to make a bot, this section is specifically intended to teach you how to make a *good* bot.
This is, of course, my personal experience as well as what I have learned from testing and speaking with other bot creators.
*If you believe some part of this section is wrong or misleading, please let me know so I can fix it!*

This section aims to provide enough general and specific guidance to allow a dedicated bot creator to train a bot from nothing to GC.

I will be referencing the [rewards section](rewards.md), [learner settings section](learner_settings.md), and [graphs section](graphs.md) frequently.

# Early Stages (Bronze - Silver)

There are differing opinions on what "early stages" mean, but personally I define the early stages of training as the period of time before your bot is actually trying to score. 
Bots in the early stages cannot yet push or shoot the ball into the goal.

In this stage, you primarily should focus your bot on these 2 tasks:
1. Learn to touch the ball
2. Don't forget how to jump

### Why do bots forget how to jump?

Controlling your car in the air is hard and is a lot less forgiving than driving on the ground.
A fresh bot simply tasked with reaching the ball will learn this very quickly, and will stop pressing jump altogether.
However, jumping is very obviously useful, and it can be very difficult for the bot to rediscover jumping later on.
To combat this, most bots use some sort of reward for being in the air, or a penalty for being on the ground.

An alternative is to add more jump actions to a discrete action parser. 
This will make a higher portion of possible actions have jumping, and will greatly increase jump usage in early stages. 
Doubling the jump actions seems to be enough to eliminate the need for air rewards or ground penalties.

### What rewards should I use in the early stages?

From my testing, here are some good rewards to get a fresh bot to learn to hit the ball as quickly as possible.
```py
// Format: (reward, weight)
rewards = (
	(EventReward(touch=1), 50), # Giant reward for actually hitting the ball
	(SpeedTowardBallReward(), 5), # Move towards the ball!
	(FaceBallReward(), 1), # Make sure we don't start driving backward at the ball
	(AirReward(), 0.15) # Make sure we don't forget how to jump
)
# NOTE: SpeedTowardBallReward and AirReward can be found in the rewards section of this guide
```

Notice how I didn't include any rewards for scoring, or even moving the ball toward the goal.
Having these rewards before the bot is capable of actually hitting the ball just adds lots of noise to the overall reward and will slow learning.

I recommend using a learning rate of around `2e-4` for the early stages.

After running these rewards for a few dozen million steps, your bot should be hitting the ball quite frequently.

If your bot stops jumping, increase the `AirReward`!

### Learning to score

Once your bot is capable of hitting the ball, you should introduce rewards for moving the ball to the goal and scoring.
You should also decrease the `TouchBallReward` a lot so that it is no longer the bot's top priority.

I recommend using `VelocityBallToGoalReward` as the continuous scoring encouragement, it should be a fair bit stronger than `SpeedTowardBallReward`.

### Don't give massive goal rewards!

Many people are inclined to add a goal reward with massive weight to the bot, like this:
```py
reward = (
	(EventReward(team_goal=1, concede=-1), 100),
	(VelocityBallToGoalReward(), 2),
	(SpeedTowardBallReward(), 1),
	(FaceBallReward(), 0.1),
	...
)
```

**Don't do this!**

This *feels* like it makes sense because goals are the most important thing in the game.
However, from my testing and experience, adding massive goal rewards early on in training simply slows down learning and decreases exploration.

A giant goal reward will drown out every other reward you have. Pick a more reasonable weight like `20`, in this instance.

I have trained a bot to element level without the use of any goal rewards. It is less important than you might expect.

# Middle Stages (Gold - Plat)

Once your bot is capable of pushing the ball into the net, you are now in the "middle stages".
This stage is more complex and more difficult to get right.

There are several different things you probably want your bot to learn in this stage:
- Basic shots
- Basic saves
- Basic jump-touches and baby aerials
- Basic 50s 
- Collecting boost and keeping
- Giving space to teammates (if your bot isn't 1v1-only)

Some of these behaviors will develop naturally if given enough time, whereas others are much harder for bots to discover on their own.

Also, you'll generally want to decrease LR to around `1e-4` now that you're out of the early stages. Watch your clip fraction!

### A better ball-touch reward

The default `touch` part of `EventReward` is not very good once your bot can touch the ball. 
This is because ball touches can easily be farmed by constantly pushing the ball, instead of shooting or cutting it.

A substantial improvement to this reward is to **scale the reward with the strength of the touch**. 
This means that a slight push that barely changes the velocity of the ball will give almost no reward, but a strong shot will give lots of reward.

I won't provide a copy-pasteable reward for this, as you should try to write this reward on your own. Here are some hints:
- `ball_touched` is a property of players, and is `True` if they hit the ball since the previous step
- You should use the ball's current velocity and previous velocity to calculate the change in velocity

### A good air-touch reward

Bots find the air scary, so they usually need some strong encouragement to actually hit the ball in the air, especially when it is up high.

A basic air touch reward just rewards the player for how high the ball is:
```py
reward = ball.position[2] / CommonValues.CEILING_Z
```

However, the bot will usually learn to just hit the ball off of the wall high-up to get this reward with minimal effort--usually in the lame "plat wall-shot" way, not the cool "air dribble" way.
A solution is to track the amount of time the player has spent in the air (or get it from RocketSim's `air_time`), and to combine that with the height scaling.

```py
MAX_TIME_IN_AIR = 1.75 # A rough estimate of the maximum reasonable aerial time
air_time_frac = min(player.air_time, MAX_TIME_IN_AIR) / MAX_TIME_IN_AIR
height_frac = ball.position[2] / CommonValues.CEILING_Z
reward = min(air_time_frac, height_frac)
```

### Get boost, and don't waste it this time!

Bots will sort of discover picking up boost if given enough time, but are generally pretty wasteful once they have it.

I always recommend a general `SaveBoostReward`, which rewards the player based on how much boost they have.
```py
reward = sqrt(player.boost_amount)
```

Note that I am using `sqrt()` here. 
The `sqrt()` effectively makes boost more important the less you have (within this reward), which is just a fact of Rocket League. 
Going from 0 boost to 50 boost is more useful than going from 50 boost to 100 boost (remember that boost ranges from 0-1 in RLGym stuff).

If your bot is wasting boost, increase the `SaveBoostReward`. If your bot is hogging boost and is afraid to use it, decrease the reward.

For picking up boost, `EventReward`'s `boost_pickup` is a good starting point. 
However, bots have a tendency to ignore small pads, so I recommend making the small pad pickup reward much stronger than just 12% of the big boost pickup reward (via a custom reward).

### Developing outplays

One of the key skills that will bring your bot into the later stages is the ability to outplay.
This means learning a mechanic that can change the direction of the ball to get past a challenging opponent.

The most common way bots do this is with dribbling and flicking, a behavior that comes quite naturally to bots. However, this is not the only way to outplay opponents. 
Cuts, passes, air dribbles, and flip resets are also very strong outplays--however most of those mechanics typically aren't discovered on their own.
If there's a particular mechanic you would like your bot to perform to outplay, I recommend creating and adding a reward for that mechanic in these stages.

# Later Stages (Diamond+)

Once your bot is capable of outplaying opponents, collecting boost, saving, shooting, and other fundamental game mechanics, I consider it to be in the later stages.
Bots entering these stages are usually around diamond rank (although its 1v1 rank may be plat, as 1v1 ranks are more difficult).

## How do I know if my bot is improving?

The better your bot gets, the slower it will improve. This is the nature of pretty much any improving thing, and bots are no exception.

In the early stages, it is blatantly obvious if your bot is improving or not. 
In the middle stages, it is less obvious, but repeated observation usually makes it clear.
However, in the later stages, it can be very difficult to tell.

Since the bot is playing against itself, an increase in reward, goals, or most other metrics do imply a change in gameplay, but not improvement.

Luckily, the PPO learning algorithm is pretty good at its job and generally doesn't get worse at a task unless you mess something up really bad.
However, the learning algorithm is trying to maximize and explore your rewards, not winning. 
So, if your rewards are farmable in a way that does not improve your bot's skill, your bot will likely get worse at the game.

### Objectively measuring improvement

The bot training framework [rocket-learn](https://github.com/Rolv-Arild/rocket-learn) uses an ELO-like rating system to track the skill of the bot against previous versions, 
so you can actually measure how much the bot is improving. I do plan on implementing such a system for rlgym-ppo in both C++ and Python, but I haven't gotten around to it yet.

If you want to manually test improvement, you can verse your current bot against an older version and see who comes out on top. 
If you are doing this in RLBot, you will have to wait a while, as it takes many goals to actually begin to know if the bot is improving.

For my first rlgym-ppo bot I wrote a little Python script that used rlgym-sim to run an infinite game between two versions of my bot. 
This could run far faster than real-time (unlike RLBot) so I was quickly able to see if the bot was actually improving as the scoreline rapidly climbed.

## Nextoification

Nexto mostly used very general and gentle rewards, and while there was definitely some deliberate influence on playstyle and mechanics, such as aerials, its a good example of how bots *want* to play.

Generally, the less specifically you reward your bot, the more its playstyle will resemble Nexto. 
Nexto's passive dribble-flick playstyle with mostly forward flicks seems to be a natural evolution of basic ballchasing behavior.

### Natural Dribbling and Flicking

Dribbling and flicking will generally be discovered in most bots with basic rewards, even without any reward for dribbling or flicking.

Almost all bots have a far faster reaction time than humans. Nexto, with a `tick_skip` of `8`, can react to something in as little as 67 milliseconds--whereas humans take around 200-300ms.
This makes dribbling far easier, as the bot just has to react to the ball falling off the edge of its car by accelerating or turning that way, instead of having to predict how the ball will move.
Dribbling is just a natural evolution of pushing the ball in that sense.

This also makes flicking over opponents far easier, as the bot can wait until the very last moment to flick, unlike humans, who have to guess or flick much easier to avoid being successfully challenged.

You can still use some slight encouragement to start dribbling if you don't want to wait for it to evolve, or if you have big rewards for doing something different (like air dribbling).

### Natural Passiveness

While Nexto's dribbles and flicks are extremely precise and effective, one of its main flaws as a bot is that it is very passive.
If you are persistent, Nexto will often give you the ball for free and wander away to its goal, applying no pressure and allowing you to set up a strong outplay with ease.
This is a flaw you will notice with most bots. They are unwilling to take risks and be aggressive when given the option. They would rather wait to save than fight to score.

I believe there are two main reasons why passiveness is so natural for bots:
1. Having a faster reaction time makes being passive more viable (you can react faster to threats)
2. It is simply easier to be passive than to be aggressive because being aggressive requires more predictive decision-making

Personally, I am not a fan of passiveness in bots. In all of my bots I have taken many steps to encourage and promote riskier, more aggressive play.
This allowed my bots to discover much stronger plays and defense as a result.

But *how* do you promote aggression? My most general solution to this problem is to **just decrease the concede penalty**.
It seems like basic logic that your reward for conceding (getting scored on) should just be the goal reward negative. After all, getting scored on is just as bad as scoring is good.
However, as you may have already learned, just because something is theoretically correct doesn't mean it is optimal for training bots.

I introduce what I call `aggression_bias`, which is the portion of the concede penalty that is removed to promote aggression. 
At `aggression_bias = 0.25`, the penalty for conceding is 25% less than the reward for scoring.
You can define the concede reward using `aggression_bias` like so: `concede_reward = -goal_reward * (1 - aggression_bias)`.

I generally use an `aggression_bias` of around `0.2` in my bots, but I frequently change it depending on how aggressive the bot is playing.
If the `aggression_bias` isn't enough, you may want to add strong rewards for challenging a play, and a penalty for there being no player on a given team near the ball.

## Letting it cook

In these later stages, it is more important to allow the bot to slowly explore and improve on its own.
Sometimes, you will see no changes on the graphs, but that often means it is slowly improving at everything instead of changing how it plays.

A good amount of patience is required to get a high-level bot.

_____
[Back to Table of Contents](README.md)
