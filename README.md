
# Making your first Rocket League ML bot using RLGym-PPO

This guide will explain how to get started with RLGym-PPO, a very nice and easy-to-use learning framework for making Rocket League bots. I will explain what all of the settings in `example.py` do, and various recommendations.

*DISCLAIMER: The recommendations I will give are based on my personal experience, as well as what I have learned from talking to and reading the code of other bot creators. I am definitely no expert, and my experience is limited to just a few bots. Furthermore, no piece of advice can apply to all bots, and while I will do my best to give general sentiments, some may not apply perfectly (or at all) to your bot. Experiment and see what works best! That's the only way to truly know.*

*Also, if you notice a mistake in this guide, let me know!*

## Prerequisites
This guide assumes you have some basic Python experience. If you are coming from another language, that's fine too, but you may need to google basic information and watch a few tutorials along the way. I won't hand-hold basic Python tasks such as adding an import or making a function, nor will I explain what an argument or constructor is. If you don't know, there is no shame in looking it up.

This guide also assumes you know the basics of Rocket League. If you don't know what Rocket League is, I have no idea how you got here.

You don't need any prior experience in machine learning, and in fact this guide will assume that you don't. If you already know a concept I explain, feel free to skip ahead.

## Installing RLGym-PPO and RLGym-Sim
Follow the instructions on https://github.com/AechPro/rlgym-ppo/blob/main/README.md.
If you have an NVIDIA GPU, you should definitely install PyTorch with GPU support, because it will greatly speed up training.

### Wait where is Rocket League involved?
RLGym-PPO uses RLGym-Sim, which is a version of RLGym that runs on a simulated version of Rocket League, called RocketSim (far more often stylized as rocketsim), without actually running the game itself. This means that you can use RLGym-PPO on non-windows platforms, without needing Rocket League, and can also collect data from the game much faster.

## Actually running your bot
Once you have installed RLGym-PPO, you can run your bot by running `example.py`.

This will start training the bot, and will report its results to *WandB* (Weights and Biases), a data platform you can use to view graphs of various statistics as your bot trains. It will also print out a large list of information into the console, which I will elaborate on in the next section.

## The basics of the training loop
Training is a process of:
 - **Collection**: The bot collects data from the environment (i.e. the bot plays the game at super-speed). Each data point during gameplay is called a **step** (or, as it will often be referred to throughout this guide, a **timestep**). More information on steps will be covered later.
 - **Learning**: The learning algorithm uses those collected steps to update the brain of the bot. Being RLGym-**PPO**, the learning algorithm is Proximal Policy Optimization (PPO).

Each time this cycle of learning repeats it is called an **iteration**. After each iteration, the bot will print out a report, which will look something like this:

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

Some of these terms are quite complicated and require a far deeper dive into ML than is within the purview of this guide, so I'll only cover the simpler ones.

`Policy Reward`: This is how much reward the bot collected, on average, in each **episode**. An **episode** is a small piece of gameplay that ends once a certain condition is met, such as a goal being scored.

`Collected Steps per Second`: This is how many **steps** of gameplay are being collected every second. The better computer you have, the higher this number will be. 

A **step** is a tiny portion of time in the game. Rocket League physics runs at 120 FPS, and each of these physics frames is called a tick. A step is a handful of ticks, 8 by default. How many ticks is within a step is called the **tickskip**. Since `120/8 = 15`, this means your bot is running at 15 steps per ingame second. On each step, the bot picks its new input.

We'll talk more about this later.

`Overall Steps per Second`: Collected steps/sec is just for the collection phase, but what about the learning phase? Overall steps/sec includes the time it takes to learn after collecting all the data. If you want to know how quickly your bot is running overall, this is the number to look at.

`Timestep Collection Time`: How long it took to collect all of the steps.

`Timestep Consumption Time`: How long it took to "consume" (organize and learn from) all of those steps.

`Cumulative Timesteps`: This is the total number of steps collected during your bot's entire lifetime. If you plan on making a good bot, expect to see numbers in the tens of billions someday!

`Timesteps Collected`: This is the number of steps collected within the iteration itself, rather than across its entire lifetime as the cumulative timesteps are.

*Fun fact: If your bot steps 15 times a second, 1 billion steps is ~18,500 hours of Rocket League. Bots do not go outside.*

## Ok, but how does it learn to play RL?

When your bot first starts learning, it has absolutely no idea what is going on. Its brain starts off completely random, so it just mashes random inputs. Also, for future reference, when I say **resetting the bot**, I mean resetting all learning back to this state of completely random play. Resetting the bot may be a good choice for a number of reasons, and usually occurs when the bot is not improving, or a significant change needs to be made that would break the current bot.

In order to get our bot to actually learn something, it needs to have **rewards**. Rewards should encourage things that happen in the game you want the bot to do more often and punish things you want it to do less often. The learning algorithm is designed to try to maximize how much reward the bot is getting, while also exploring new ways to get even more rewards.

Rewards can be both positive and negative, and are ultimately just numbers assigned to specific steps. Positive rewards encourage behaviors while negative rewards punish them.

By default, your bot has 3 rewards:
 - `VelocityPlayerToBallReward()`: Positive reward for moving towards the ball, negative reward for moving away
 - `VelocityBallToGoalReward()`: Positive reward for having the ball move towards the opponent's goal, negative reward for having the ball move towards your own goal
 - `EventReward()`: Positive reward for scoring, negative reward for conceding (getting scored on), and a smaller reward for getting a demo

*You can find these rewards in `example.py`.*

As your bot continuous to mash random inputs, it will accidentally trigger a reward or punishment. Then, during the learning phase, the learning algorithm will try to adjust the bot's brain such that it is more or less likely to do the things that were rewarded or punished.

### Wait, why don't we just have a goal reward?
If you think about it, the only thing that ultimately matters for winning a game of Rocket League is scoring and not getting scored on, so why do we need other rewards?

Well, the answer is that bots are not very smart compared to humans. They can't plan out what they want to do, nor do they know what a car is, nor have they ever heard of balls or goals before. They need a lot more specific encouragement to learn how to move towards the ball, hit the ball, collect boost, and so on. 

A lot of the difficulty of making a bot is creating rewards that encourage the bot to do what you want without limiting its ability to explore other options.

## Diving into actually modifying our bot

So, now that you know some of the fundamental ideas, let's start actually messing with stuff.

If you haven't already, open `example.py` in the Python editor of your choice.

### Action parser

The first thing I recommend changing is the action parser.
This is set at the line:

```python
action_parser = ContinuousAction()
```

**Continuous actions** mean the bot can use any partial input, which allows for more precise input. However, this is more difficult to use, and I do not recommend it as your first action parser.

An **action** is the combination of controller inputs the bot presses (throttle, steer, jump, boost, etc.), and an action parser converts the outputs of the bot's brain into these controller inputs.

Most bots use a **discrete action parser**, which separates every useful permutation of inputs into their own box, and the bot can control the car by picking a specific box of inputs.

Now, before you go ahead and swap out `ContinuousAction` with `DiscreteAction`, beware:
`DiscreteAction` is actually `MultiDiscrete`, which is not what I described.
The fully-discrete[*] action parser is called `LookupAction`, and is not included by the library by default.

You can find it here: https://github.com/RLGym/rlgym-tools/blob/main/rlgym_tools/extra_action_parsers/lookup_act.py

Since action parsers define how your bot controls the car, changing it usually means resetting the bot.


### Rewards and weights

Down below the list of rewards is a list of numbers.
These numbers are the weights of each reward, which is how intensely they will influence the bot. `VelocityPlayerToBallReward()` is the lowest, `VelocityBallToGoalReward()` is 10x more influential, and `EventReward()` is 100x more influential than that.

**Event rewards** are rewards that activate once, and only when a specific thing happens. They are usually important game events, like hitting the ball, shooting, scoring, and so on.

**Continuous rewards** are rewards that are active *while* something is happening, and thus can run for many steps in a row. Since they happen so often, they are inherently stronger than event rewards, and usually should have far less weight.

`VelocityPlayerToBallReward` and `VelocityBallToGoalReward` are continuous rewards, whereas `EventReward` is... well, it's in the name.

Inside the constructor to `EventReward()` are sub-weights for different events. Each event will be multiplied by its sub-weight, then multiplied by the reward's weight after. Scoring is `1 * 10 = 10` reward total, whereas demos are only a tenth the reward of scaling, `0.1 * 10 = 1` reward total.

All rewards are eventually normalized in the learning algorithm (unless you specifically turn that off, which you probably shouldn't). This means that what actually matters is how rewards are weighted *in relation* to other rewards.

I recommend that you:
- Increase `VelocityPlayerToBallReward` a bit (it's very important in the early stages)
- Add a `FaceBallReward` with a small weight (this will reward your bot for facing the ball, which is very helpful during early learning)

### Obs builder

**Obs** is short for observation, and it is how your bot perceives the game. An **obs builder** converts the current **state** of the game into inputs to your bot's brain.

The default obs builder is a decent starting point. I've found you can get moderately better results if you add car-relative positions and velocities, but that's a bit more advanced.

Making obs builders is quite tedious, and also very very high-risk. If you slightly mess something up, it could make your bot unable to play the game, or even worse, play the game but poorly (a very insidious genre of bug, because it can be very difficult to tell that something is wrong).

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

These are a set of conditions that define when episodes end. If any of these conditions trigger, the episode is over. Nearly all bots have a `GoalScoredCondition` (because otherwise the ball is going to stay inside the goal... which is weird), as well as a `NoTouchTimeoutCondition`.

The `NoTouchTimeoutCondition` ends the episode if no player has touched the ball for a certain amount of time (10 seconds by default). This is helpful, especially in the early stages, and prevents you from wasting time collecting a ton of data on two motionless bots who aren't doing anything or are stuck upside-down.

### State setters

Once the game is terminal, it needs to be reset. By default, it will be reset to kickoff.

However, for beginner bots, kickoff is usually not the best state setter. 

I recommend using the `RandomState` state setter, especially in the early stages of training. 
Its constructor takes 3 arguments: `ball_rand_speed`, `cars_rand_speed`, and `cars_on_ground`. 
I recommend you use `(True, True, False)`, as this will make the cars and ball start at a random location with a random velocity. 
The cars will also spawn airborne half of the time, meaning they will quickly learn how to orient themselves in the air with some level of competency.

The state setter is an argument of `rlgym_sim.make()` within your `build_rocketsim_env()`.

## Next Sections:

[Learner Settings](learner_settings.md) <- *What the various learner settings do*

[Rewards](rewards.md) <- *How rewards work, and how to write them*

[Visualizing Your Bot](visualizing.md) <- *How to watch your bot play in a visualizer*

[Understanding The Graphs](graphs.md) <- *What the metric graphs mean*

### *TODO: Add more info on metrics, etc.*
