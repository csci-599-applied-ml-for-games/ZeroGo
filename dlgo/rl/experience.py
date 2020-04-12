import numpy as np

__all__ = [
    'ExperienceCollector',
    'ExperienceBuffer',
    'combine_experience',
    'load_experience',
]


class ExperienceCollector:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.advantages = []
        self.stone_advantages = []
        self.value_rewards = []
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []
        self._current_stone_advantages = []

    def begin_episode(self):
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []
        self._current_stone_advantages = []

    def record_decision(self, state, action, estimated_value=0, stone_advantage=0):
        self._current_episode_states.append(state)
        self._current_episode_actions.append(action)
        self._current_episode_estimated_values.append(estimated_value)
        self._current_stone_advantages.append(stone_advantage)

    def complete_episode(self, reward):
        num_states = len(self._current_episode_states)
        self.states += self._current_episode_states
        self.actions += self._current_episode_actions
        self.stone_advantages += self._current_stone_advantages
        self.rewards += [reward for _ in range(num_states)]

        for i in range(num_states):
            stone_advantage = self._current_stone_advantages[i]  # current advantage on board
            stone_advantage = np.clip(stone_advantage, -10, 10) * 0.1  # clip to [-1, 1]

            if i <= 15:
                discount_reward = 0  # no reward for the first 30 steps (15 for each)
            elif i < 105:
                discount_reward = reward * \
                (i - 15) ** 2 / (90 ** 2)  # reward discounted rate (i - 15) ** 2 / (135 - 15) ** 2
            else: 
                discount_reward = reward  # full reward after 270th step

            # adjust reward according to stone advantage
            if (stone_advantage == 1 and discount_reward < 0) or \
                (stone_advantage == -1 and discount_reward > 0):
                discount_reward *= 0.25
            elif stone_advantage * discount_reward < 0:
                discount_reward = 0.25 * discount_reward + \
                    0.75 * discount_reward * (1 - abs(stone_advantage))

            # value_reward = (np.mean([stone_advantage, discount_reward]) + 1) / 2
            # scale to [0, 1]
            value_reward = (discount_reward + 1) / 2
            self.value_rewards.append(value_reward)

            advantage = reward - self._current_episode_estimated_values[i]
            self.advantages.append(advantage)

        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []
        self._current_stone_advantages = []


class ExperienceBuffer:
    def __init__(self, states, actions, rewards, advantages, stone_advantages, value_rewards):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.advantages = advantages
        self.stone_advantages = stone_advantages
        self.value_rewards = value_rewards

    def serialize(self, h5file):
        h5file.create_group('experience')
        h5file['experience'].create_dataset('states', data=self.states)
        h5file['experience'].create_dataset('actions', data=self.actions)
        h5file['experience'].create_dataset('rewards', data=self.rewards)
        h5file['experience'].create_dataset('advantages', data=self.advantages)
        h5file['experience'].create_dataset('stone_advantages', data=self.stone_advantages)
        h5file['experience'].create_dataset('value_rewards', data=self.value_rewards)


def combine_experience(collectors):
    combined_states = np.concatenate([np.array(c.states) for c in collectors])
    combined_actions = np.concatenate([np.array(c.actions) for c in collectors])
    combined_rewards = np.concatenate([np.array(c.rewards) for c in collectors])
    combined_advantages = np.concatenate([np.array(c.advantages) for c in collectors])
    combined_stone_advantages = np.concatenate([np.array(c.stone_advantages) for c in collectors])
    combined_value_rewards = np.concatenate([np.array(c.value_rewards) for c in collectors])

    return ExperienceBuffer(
        combined_states,
        combined_actions,
        combined_rewards,
        combined_advantages,
        combined_stone_advantages,
        combined_value_rewards)


def load_experience(h5file):
    return ExperienceBuffer(
        states=np.array(h5file['experience']['states']),
        actions=np.array(h5file['experience']['actions']),
        rewards=np.array(h5file['experience']['rewards']),
        advantages=np.array(h5file['experience']['advantages']),
        stone_advantages = np.array(h5file['experience']['stone_advantages']),
        value_rewards=np.array(h5file['experience']['value_rewards']))
