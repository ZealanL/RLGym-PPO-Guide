# Learner settings

Next, we will cover most of the settings for the `Learner`. The `Learner` is a Python class that runs all of the learning loop, and it has many settings for all sorts of things related to the learning process.

Learner settings are set in the `Learner` constructor in `example.py`.
```py
learner = Learner(build_rocketsim_env,
                  n_proc=n_proc,
                  min_inference_size=min_inference_size,
                  metrics_logger=metrics_logger,
                  ...
```
⚠️ *Many of the settings set in `example.py` are bad, such as `ent_coef`. Please read about them and change them.*

Note that only some of the learner settings are being set here. To see all settings, go to [rlgym-ppo/learner.py](https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/learner.py) and look at the constructor (`def __init__(...`).

___
`n_proc`: To make learning faster, multiple games are run simultaneously, each in its own Python process. This number controls how many Python processes are launched. I recommend increasing this number until you have max CPU usage, to get the best possible steps/second (you can check in Task Manager on Windows).
___
`render`: This enables render mode, which slows down one of the games and sends it to a rendering program. By default, RLGym-PPO uses https://github.com/VirxEC/rlviser/.

You will turn this on when you want to watch your bot play, but don't leave it on, as it will slow down the learning.
___
`render_delay`: The delay of sending steps to the renderer. Lowering this speeds up the game speed in render mode. I like to have this lower than real-time so I can watch the bot in ~2x speed.
___
`timestep_limit`: How many timesteps until the learner automatically stops learning. I like to set this to a stupid big number, like `10e15` (1 quadrillion). 
___
`exp_buffer_size`: The learning phase doesn't just learn based on the most recent collected batch of steps, but also the previous few batches. This setting controls how big the buffer that stores all the steps is. I recommend setting this to `ts_per_iteration*2` or `ts_per_iteration*3`.
___
`ts_per_iteration`: How many steps are collected each iteration. The optimal number is highly dependent on a number of factors, and you can play around with it to see what results in faster learning. I've found that values of `50_000` are good for early learning, but once the bot is actually hitting the ball, it should be increased to `100_000`. Once the bot is actually shooting and scoring, increase it to `200_000` or even `300_000`.
___
`policy_layer_sizes`: How big your bot's **policy** is. 

Each number is the width of a layer of neurons. By default, there are 3 layers, each with 256 neurons.

I haven't mentioned this yet for simplicity, but the learning algorithm actually uses two neural networks: a **policy** that actually plays the game, and a **critic** that predicts how much reward the policy will get.

The critic learns to predict the reward the policy will get in a given situation, and the policy learns to get reward that the critic didn't predict. This causes the bot to explore methods of new getting reward, resulting in better learning.

The default policy size is quite small, I would highly recommend increasing both policy and critic size until you start losing a ton of SPS (steps per second). In general, a bigger policy and critic will learn better.

My computer (with a RTX 3060 ti) seems to run best on a policy/critic size of `[2048, 2048, 1024, 1024]`. More than that, and my SPS tanks. If you aren't using a GPU, your CPU is going to take ages to run a large network, so you might need to stick with a smaller one.

If you change this, you will need to reset your bot.
___
`critic_layer_sizes`: This is usually set to the same sizes as the policy.

There is some evidence to suggest that the critic should be bigger, but this hasn't been thoroughly tested in Rocket League yet.

I recommend just making it the same size as the policy, unless you know what you are doing.

If you change this, you need to reset the critic. However, the critic doesn't play the game, so you can train a new critic on the same policy. You should set the policy's learning rate to 0 while you do this, so the noob critic doesn't screw up the policy while it is still figuring out what's happening.
___
`ppo_epochs`: This is how many times the learning phase is repeated on the same batch of data. I recommend 2 or 3. 

Play around with this and see what learns the fastest. Increasing this will lower your SPS because the learning step is repeating multiple times, but you will get better learning up until a certain point. When testing, compare the increase in rewards from a common starting point, not SPS.
___
`ppo_batch_size`: Just set this to `ts_per_iteration`. This is the amount of data that the learning algorithm trains on each iteration.
___
`ppo_minibatch_size`: This should be a small portion of `ppo_batch_size`, I recommend `25_000` or `50_000`. Data will be split into chunks of these size to conserve VRAM (your GPU's memory). This does not affect anything other than how fast the learning phase runs. Mess around and see what gives you the highest SPS.

If you aren't using a GPU, this isn't as important. I have no clue what the optimal value is for CPU learning. I'd guess something very big (RAM is usually bigger than VRAM), or something quite small (CPU cache size).
___
`ent_coef`: This is the scale factor for entropy. Entropy fights against learning to make your bot pick actions more randomly. This is useful because it forces the bot to try a larger variety of actions in all situations, which leads to better exploration.

The golden value for this seems to be about `0.01`. 
You can reduce this significantly to cause your bot to stop exploring the game and start refining what it already knows. Don't do this if you plan to continue training your bot after.
___
`policy_lr`/`critic_lr`: The learning rate for the policy and critic. If you have experience in supervised ML, this means the same thing. I recommend keeping them the same unless you know what you're doing.

Bigger values increase how much the policy and critic change during learning. 
Too small, and you are wasting time. Too big, and the learning gets stuck jittering between different directions, unable to actually progress.

I generally start LR high, then slowly decrease it as the bot improves. If your bot seems stuck, try decreasing LR. Generally, the better the bot is, the lower the LR should be. 

Here's what I've found to be ideal at different stages of learning:
- Bot that can't score yet: `2e-4`
- Bot that is actually trying to score on its opponent: `1e-4`
- Bot that is learning outplay mechanics (dribbling and flicking, air dribbles, etc.): `0.8e-4` or lower

The optimal values for your bot will be different, though. This is a value you should play around with until you find something that seems optimal.
___
`log_to_wandb`: Whether or not to log metrics to `wandb`.
___
`checkpoints_save_folder`: Where to save checkpoints to.

Checkpoints are your bot's current policy and critic, as well as some associated info.

Checkpoints are used to save and load the bot.

If you mess something up and your bot's brain gets fried, you will want to restore to an earlier checkpoint.

I also save backups every day or so, because only so many checkpoints are stored, and sometimes all of them are fried without you realizing until after.
___
`checkpoint_load_folder`: The folder to load a SPECIFIC checkpoint from.

This is not set by default, meaning the bot will not automatically re-load.

It is easy to assume this will just load the most recent checkpoint, but no, it loads a specifically chosen checkpoint. 

I recommend you add a little code to load the most recent one and save yourself the effort.

A neat little Python line for getting the name of the most recent checkpoint in a folder (written by Lamp I think?):
```py
# Note: You MUST disable the "add_unix_timestamp" learner setting for this to work properly
latest_checkpoint_dir = "data/checkpoints/rlgym-ppo-run/" + str(max(os.listdir("data/checkpoints/rlgym-ppo-run"), key=lambda d: int(d)))
````
___	
[Back to Table of Contents](README.md)
