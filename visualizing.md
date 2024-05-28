# Visualizing your bot

This short section will explain how to watch your bot play the game.

Since we are training in `rlgym_sim` which uses RocketSim, a simulated version of Rocket League, the actual game isn't running.
So, if we actually want to watch our bot, we need a program that can render gamestates in 3D so we can watch what is going on.

Such programs for rendering RocketSim games are called **visualizers**.

There are multiple visualizers to choose from, the default one is VirxEC's https://github.com/VirxEC/rlviser/

Virx's visualizer sometimes can be troublesome to set up, so I wrote my own visualizer with the goal of being as easy to use as possible.
You can find it, along with how to connect it to RLGym-PPO/`rlgym_sim`, here: https://github.com/ZealanL/RocketSimVis

Regardless of what renderer you use, you can enable rendering and adjust the render delay in [RLGym-PPO's learner settings](https://github.com/ZealanL/RLGym-PPO-Guide/blob/main/learner_settings.md).

By default, the render delay is very small, so you will be watching your bot in 2x speed or whatever.

To determine the render delay for normal speed, you should use the knowledge that:
- A state is sent to the renderer each step
- Each step is `tick_skip` ticks
- There are 120 ticks in a second