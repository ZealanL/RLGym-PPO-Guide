# Understanding the graphs

> :warning: *This section is a W.I.P. and needs more review!* :warning:

This section will explain to you what most of the default RLGym-PPO metric graphs mean, and how to interpret them.

By default, RLGym-PPO logs metrics with wandb. If you have wandb enabled, you can view them at your bot's wandb link, which is posted in the console when you start a training session.

Note that I will purposefully not elaborate on some of the more complicated stuff. I may add optional more-detailed explanations in the future.

## *Remember: Watch your bot!*

Unless you have an ELO-tracking system (where different versions of your bot are versed against each other to determine skill), no graph is going to be as good as of an indicator of progress as watching your bot play.

You shouldn't use vague graphs (like policy reward) to come to vague conclusions. Obsessing over weird changes and patterns in graphs like entropy or clip fraction will drive you insane and just waste your time. Totally normal learning often results in weird cyclic behavior in some graphs. Sometimes graphs suddenly shift for seemingly no reason at all.

However, if a general graph completely dies or skyrockets to an insane value, something *is* probably broken. Just don't solely rely on the graphs to determine if there is a problem unless it's very obvious.

## Policy Reward
![image](https://github.com/ZealanL/RLGym-PPO-Guide/assets/36944229/cb480e81-38c4-488e-9b2a-f563257ef7ca)

This graph shows the average of the total reward each player gets, per episode.

This should increase a lot in the early stages, HOWEVER, please DO NOT assume that a decrease or plateau in this graph means your bot is not improving.

The average episode reward is directly scaled by the length of each episode. 
At the beginning stages, your episodes are probably ending due to the timeout condition, so the average episode duration will increase if your bot starts hitting the ball.
However, as your bot starts hitting the ball, the goal-scored condition will become the primary episode-ender. Therefore, the more often your bot is scoring, the shorter the episodes, and the lower the "policy reward".

From my experience, this graph will increase very strongly at the early stages of learning.
Then, once the bot can hit the ball frequently, it will begin to flatten out or decrease, as goal scoring becomes prominent.

Note that if you are using zero-sum rewards, this graph is basically useless, as the average total zero-sum reward over any period of time is just going to be zero.
However, since zero-sum rewards are not helpful when the bots can't really reach the ball yet, this is still useful for tracking the progress of early learning.

## Policy Entropy
![image](https://github.com/ZealanL/RLGym-PPO-Guide/assets/36944229/a3974e70-30ae-4cb2-9cf6-3fad02a09fb3)

This one's pretty cool. It shows how much variety there is in the actions of your bot, on average. This graph will directly scale with `ent_coef`, as well as what situations the bot is in.

## Value Function Loss
![image](https://github.com/ZealanL/RLGym-PPO-Guide/assets/36944229/832d5b31-3bef-4551-ad8e-8dfed9d90d45)

This graph shows how much the critic is struggling to predict the rewards of the policy. 
The graph scales pretty consistently with how often event rewards (goals, demos, etc.) occur, as they are extremely difficult (and in many cases straight-up impossible) for the critic to predict in the future.

It should decrease a lot at the very beginning, but then settle down to a "best guess" at some base loss, which will then fluctuate depending on what rewards your bot is getting.

You will also see this graph immediately shift if you make significant changes to your rewards.

## Policy Update Magnitude/Value Function Update Magnitude
![image](https://github.com/ZealanL/RLGym-PPO-Guide/assets/36944229/d8726d44-dd0b-42a4-92dc-c0695adb03b7)

These are the scale of changes made to the policy and critic each iteration. These directly scale with learning rate, and they both tend to immediately spike or shift with significant reward changes.

## SB3 Clip Fraction/Mean KL Divergence
![image](https://github.com/ZealanL/RLGym-PPO-Guide/assets/36944229/ebca338e-bd0b-407a-b5cc-3f4539a54301)

These scale with the change in the policy each iteration (see "Policy Update Magnitude"). 
Many bot creators will adjust learning rate to keep one of these graphs near a certain value (I usually see people targeting ~0.08 as their clip fraction).

### *TODO: Add more graphs!*

_____
[Back to Table of Contents](README.md)