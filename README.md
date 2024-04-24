
# Making your first Rocket League ML bot using RLGym-PPO

This guide will explain how to get started with RLGym-PPO, a very nice and easy-to-use learning framework for making Rocket League bots. I will explain what all of the settings in `example.py` do, and various recommendations.

*DISCLAIMER: The recommendations I will give are based on of my personal experience, as well as what I have learned from talking to and reading the code of other bot creators. I am definitely no expert, and my experience is limited to just a few bots, so some things might not apply as well to yours. Experiment and see what works best! That's usually the only way to actually know.*

*If you notice a mistake in this guide, let me know!*

## Installing RLGym-PPO and rlgym-sim:
Follow the instructions on https://github.com/AechPro/rlgym-ppo/blob/main/README.md.
If you have an NVIDIA GPU, you should definitely install PyTorch with GPU support, because it will greatly speed up training.

### Wait where is Rocket League involved?
RLGym-PPO uses rlgym-sim, which is a version of RLGym that runs on a simulated version of Rocket League, without actually running the game itself. This means that you can use RLGym-PPO on non-windows platforms, without having Rocket League, and can also collect data from the game much faster.

## Actually running your bot
Once you have installed RLGym-PPO, you can run your bot by running `example.py`.

This will start training the bot, and will report its results to *wandb*, a data platform that you can use to see graphs of various info as your bot trains. It will also print out a big list of stuff into the console, which I will elaborate on more in the next section.

## The basics of the training loop
Training is a process of:
 - **Collection**: The bot collects data from the environment (i.e. the bot plays the game at super-speed). Each data point during gameplay is called a **step**.
 - **Learning**: The learning algorithm uses those collected steps to update the brain of the bot

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

`Timesteps Collected`: This is the total number of steps collected during your bot's lifetime. If you plan on making a good bot, expect to see numbers in the tens of billions someday!

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

## Learner settings

Next, we will cover most of the settings for the `Learner`. The `Learner` is a Python class that runs all of the learning loop, and it has a ton of settings for all sorts of things related to the learning process.

Some of these settings are already being overridden in `example.py`, but many of them are not. To see all of them, open up `rlgym-ppo/learner.py` and look at the constructor (`def __init__(...`).

`n_proc`: To make learning faster, multiple games are run simultaneously, each in its own Python process. This number controls how many Python processes are launched. I recommend increasing this number until you have max CPU usage, to get the best possible steps/second (you can check in Task Manager on Windows).

`render`: This enables render mode, which slows down one of the games and sends it to a rendering program. By default, RLGym-PPO uses https://github.com/VirxEC/rlviser/.

You will turn this on when you want to watch your bot play, but don't leave it on, as it will slow down the learning.

`render_delay`: The delay of sending steps to the renderer. Lowering this speeds up the game speed in render mode. I like to have this lower than real-time so I can watch the bot in ~2x speed.

`timestep_limit`: How many timesteps until the learner automatically stops learning. I like to set this to a stupid big number, like `10e15` (1 quadrillion). 

`exp_buffer_size`: The learning phase doesn't just learn based on the most recent collected batch of steps, but also the previous few batches. This setting controls how big the buffer that stores all the steps is. I recommend setting this to `ts_per_iteration*2` or `ts_per_iteration*3`.

`ts_per_iteration`: How many steps are collected each iteration. The optimal number is highly dependent on a number of factors, and you can play around with it to see what results in faster learning. I've found that values of `50_000` are good for early learning, but once the bot is actually hitting the ball, it should be increased to `100_000`. Once the bot is actually shooting and scoring, increase it to `200_000` or even `300_000`.

`policy_layer_sizes`: How big your bot's **policy** is. 

Each number is the width of a layer of neurons. By default, there are 3 layers, each with 256 neurons.

I haven't mentioned this yet for simplicity, but the learning algorithm actually uses two neural networks: a **policy** that actually plays the game, and a **critic** that predicts how much reward the policy will get.

The critic learns to predict the reward the policy will get in a given situation, and the policy learns to get reward that the critic didn't predict. This causes the bot to explore methods of new getting reward, resulting in better learning.

The default policy size is quite small, I would highly recommend increasing both policy and critic size until you start losing a ton of SPS (steps per second). In general, a bigger policy and critic will learn better.

My computer (with a GTX 3060 ti) seems to run best on a policy/critic size of `[2048, 2048, 1024, 1024]`. More than that, and my SPS tanks. If you aren't using a GPU, your CPU is going to take ages to run a large network, so you might need to stick with a smaller one.

If you change this, you will need to reset your bot.

`critic_layer_sizes`: This is usually set to the same sizes as the policy.

There is some evidence to suggest that the critic should be bigger, but this hasn't been thoroughly tested in Rocket League yet.

I recommend just making it the same size as the policy, unless you know what you are doing.

If you change this, you need to reset the critic. However, the critic doesn't play the game, so you can train a new critic on the same policy. You should set the policy's learning rate to 0 while you do this, so the noob critic doesn't screw up the policy while it is still figuring out what's happening.

`ppo_epochs`: This is how many times the learning phase is repeated on the same batch of data. I recommend 2 or 3. 

Play around with this and see what learns the fastest. Increasing this will lower your SPS because the learning step is repeating multiple times, but you will get better learning up until a certain point. When testing, compare the increase in rewards from a common starting point, not SPS.

`ppo_batch_size`: Just set this to `ts_per_iteration`. This is the amount of data that the learning algorithm trains on each iteration.

`ppo_minibatch_size`: This should be a small portion of `ppo_batch_size`, I recommend `25_000` or `50_000`. Data will be split into chunks of these size to conserve VRAM (your GPU's memory). This does not affect anything other than how fast the learning phase runs. Mess around and see what gives you the highest SPS.

If you aren't using a GPU, this isn't as important. I have no clue what the optimal value is for CPU learning. I'd guess something very big (RAM is usually bigger than VRAM), or something quite small (CPU cache size).

`ent_coef`: This is the scale factor for entropy. Entropy fights against learning to make your bot pick actions more randomly. This is useful because it forces the bot to try a larger variety of actions in all situations, which leads to better exploration.

The golden value for this seems to be near `0.01`.

`policy_lr`/`critic_lr`: The learning rate for the policy and critic. If you have experience in supervised ML, this means the same thing. I recommend keeping them the same unless you know what you're doing.

Bigger values increase how much the policy and critic change during learning. 
Too small, and you are wasting time. Too big, and the learning gets stuck jittering between different directions, unable to actually progress.

I generally start LR high, then slowly decrease it as the bot improves. If your bot seems stuck, try decreasing LR. Generally, the better the bot is, the lower the LR should be. 

Here's what I've found to be ideal at different stages of learning:
- Brand new bot: `7e-4`
- Bot that can chase the ball around and hit it: `3e-4`
- Bot that is actually trying to score on its opponent: `2e-4`
- Bot that is learning outplay mechanics (dribbling and flicking, air dribbles, etc.): `1e-4` or lower

The optimal values for your bot will be different, though. This is a value you should play around with until you find something that seems optimal.

`log_to_wandb`: Whether or not to log metrics to `wandb`.

`checkpoints_save_folder`: Where to save checkpoints to.

Checkpoints are your bot's current policy and critic, as well as some associated info.

Checkpoints are used to save and load the bot.

If you mess something up and your bot's brain gets fried, you will want to restore to an earlier checkpoint.

I also save backups every day or so, because only so many checkpoints are stored, and sometimes all of them are fried without you realizing until after.

`checkpoint_load_folder`: The folder to load a SPECIFIC checkpoint from.

This is not set by default, meaning the bot will not automatically re-load.

It is easy to assume this will just load the most recent checkpoint, but no, it loads a specifically-chosen checkpoint. 

I recommend you add a little code to load the most recent one, and save yourself the effort.

A neat little Python line for getting the name of the most recent checkpoint in a folder (written by Lamp I think?):
```py
checkpoint_load_dir = "data/checkpoints/" + str(max(os.listdir("data/checkpoints"), key=lambda d: int(d)))
````

## TODO: Add more info on metrics, creating rewards, etc. etc.
