# My Bot Stopped Improving!

This section tries to help troubleshoot bots that have stopped improving after a certain point.

## Wait. How do you know it's not improving?

Sometimes bots progress very slightly in many ways at once, instead of improving blatantly at just 1 or 2 things.
This kind of overall progress is very good, but also very hard to notice when you are just watching your bot through a visualizer now and again.

Using an ELO system or letting different versions battle it out in RLBot is a good way to actually check improvement.
Keep in mind that you need many RLBot games to actually know if a bot is improving. Sometimes equal bots can produce seemingly definitive scorelines like "10-2" or "6-0".
I recommend ignoring scorelines in RLBot and just looking at gameplay and decision making.

## 1. Your rewards

This is usually the reason bots stop improving, because it is the most fundemental part of the bot's improvement.
If your rewards encourage bad gameplay, the bot will play bad.

I recommend printing each reward in console every step (using a custom `CombinedReward`) while you watch your bot play in the visualizer.
This way, you can see exactly what rewards your bot is getting as it plays the game. I have found many bugged rewards this way.

## 2. Need more batches!

Increasing batch size can really help a bot to ascend to a higher level of consistency and skill.
I've used batch sizes as high as 750k in a GC bot to improve skill. 
Most bots don't benefit from a batch size that high, but it is a good example how high batch size can go.

If your bot is out of the early stages (see [Making a Good Bot](making_a_good_bot.md)), try increasing batch size if the bot seems to have stopped improving.
Too high batch size is just unhelpful and wastes time. If increasing your batch size doesn't help your bot, you should probably revert it back to not be wasteful.

## 2. Decrease learning rates

Too-high learning rates can cause bots to get stuck after a certain point, or just make improvement very slow.
If your learning rate is over 1e-4, try decreasing it a bit (for both policy and critic).

## 3. Increase GAE gamma

Gamma is explained in [Making a Good Bot](making_a_good_bot.md). 
Increasing it can lead to smarter gameplay, and generally a better bot, but high gamma can slow training and make it more difficult for bots to recognize new rewards.
Use with caution!

## 4. Reduce or remove tuning rewards

Tuning rewards are rewards for playing a certain way, such as a speed reward or a positioning reward.
These rewards can be very helpful for getting the bot to play the game in the way you want, but can prevent bots from improving once they reach a higher skill level.

I recommend decreasing your tuning rewards if your bot has stopped improving, or even removing them completely.

___	
[Back to Table of Contents](README.md)