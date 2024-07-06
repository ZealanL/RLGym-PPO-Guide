import numpy as np
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData

'''
This is a wrapper to put around an existing reward
Usage: ZeroSumReward(YourOtherReward(), team_spirit, [optional: opp_scale])
NOTE: Due to limitations in rlgym-sim, "previous_action" is not supported and will be "None" for the child reward
'''
class ZeroSumReward(RewardFunction):
    '''
    child_reward: The underlying reward function
    team_spirit: How much to share this reward with teammates (0-1)
    opp_scale: How to scale the penalty we get for the opponents getting this reward (usually 1)
    '''
    def __init__(self, child_reward: RewardFunction, team_spirit, opp_scale = 1.0):
        self.child_reward = child_reward # type: RewardFunction
        self.team_spirit = team_spirit
        self.opp_scale = opp_scale

        self._update_next = True
        self._rewards_cache = {}

    def reset(self, initial_state: GameState):
        self.child_reward.reset(initial_state)

    def pre_step(self, state: GameState):
        self.child_reward.pre_step(state)

        # Mark the next get_reward call as being the first reward call of the step
        self._update_next = True

    def update(self, state: GameState, is_final):
        self._rewards_cache = {}

        '''
        Each player's reward is calculated using this equation:
        reward = individual_rewards * (1-team_spirit) + avg_team_reward * team_spirit - avg_opp_reward * opp_scale
        '''

        # Get the individual rewards from each player while also adding them to that team's reward list
        individual_rewards = {}
        team_reward_lists = [ [], [] ]
        for player in state.players:
            if is_final:
                reward = self.child_reward.get_final_reward(player, state, None)
            else:
                reward = self.child_reward.get_reward(player, state, None)
            individual_rewards[player.car_id] = reward
            team_reward_lists[int(player.team_num)].append(reward)

        # If a team has no players, add a single 0 to their team rewards so the average doesn't break
        for i in range(2):
            if len(team_reward_lists[i]) == 0:
                team_reward_lists[i].append(0)

        # Turn the team-sorted reward lists into averages for each time
        # Example:
        #    Before: team_rewards = [ [1, 3], [4, 8] ]
        #    After:  team_rewards = [ 2, 6 ]
        team_rewards = np.average(team_reward_lists, 1)

        # Compute and cache:
        # reward = individual_rewards * (1-team_spirit)
        #          + avg_team_reward * team_spirit
        #          - avg_opp_reward * opp_scale
        for player in state.players:
            self._rewards_cache[player.car_id] = (
                    individual_rewards[player.car_id] * (1 - self.team_spirit)
                    + team_rewards[int(player.team_num)] * self.team_spirit
                    - team_rewards[1 - int(player.team_num)] * self.opp_scale
            )

    '''
    I made get_reward and get_final_reward both call get_reward_multi, using the "is_final" argument to distinguish
    Otherwise I would have to rewrite this function for final rewards, which is lame
    '''
    def get_reward_multi(self, player: PlayerData, state: GameState, previous_action: np.ndarray, is_final) -> float:
        # If this is the first get_reward call this step, we need to update the rewards for all players
        if self._update_next:
            self.update(state, is_final)
            self._update_next = False
        return self._rewards_cache[player.car_id]

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.get_reward_multi(player, state, previous_action, False)

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.get_reward_multi(player, state, previous_action, True)