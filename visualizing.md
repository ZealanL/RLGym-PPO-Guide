# Visualizing your bot

This short section will explain how to watch your bot play the game.

## What's a visualizer?

RLGym-PPO uses rlgym-sim, which simulates Rocket League games using uses RocketSim, the actual game isn't running.
So, if we want to watch our bot, we need a program that can render Rocket League gamestates in 3D.

Such programs for rendering RocketSim games are called **visualizers**.

There are multiple visualizers to choose from, the default one is VirxEC's https://github.com/VirxEC/rlviser/

Virx's visualizer can sometimes be troublesome to set up, so I wrote my own visualizer with the goal of being as easy to use as possible.
You can find it, along with how to connect it to RLGym-PPO/`rlgym_sim`, here: https://github.com/ZealanL/RocketSimVis

Regardless of what renderer you use, you can enable rendering and adjust the render delay in [RLGym-PPO's learner settings](https://github.com/ZealanL/RLGym-PPO-Guide/blob/main/learner_settings.md).

## Adjusting render delay

By default, the render delay is very small, so you will be watching your bot in 2x speed or whatever.

To determine the render delay for normal speed, you should use the knowledge that:
- A state is sent to the renderer each step
- Each step is `tick_skip` ticks
- There are 120 ticks in a second

I recommend defining a constant for `TICK_SKIP`, and another constant called `STEP_TIME`, which is the time between steps, written in terms of `TICK_SKIP`.

This constant will also be useful when tracking time in rewards and terminal conditions and such.
