import numpy as np
from collections import deque


class PerPromptStatTracker:
    def __init__(self, buffer_size, min_count):
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.stats = {} #* 각 prompt를 key로 해서 value에는 reward들 deque.

    def update(self, prompts, rewards):
        prompts = np.array(prompts)
        rewards = np.array(rewards)
        unique = np.unique(prompts) #* 받은 prompts들 중 unique하게 만듦
        advantages = np.empty_like(rewards)
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt] #* prompt와 같은 것만 index로 뽑아옴.
            if prompt not in self.stats:
                self.stats[prompt] = deque(maxlen=self.buffer_size)
            self.stats[prompt].extend(prompt_rewards)

            #* 각 prompt의 reward가 충분하지 않으면, 통계가 불안정하므로, 전체 reward로 standardization.
            if len(self.stats[prompt]) < self.min_count: 
                mean = np.mean(rewards)
                std = np.std(rewards) + 1e-6
            else:
                mean = np.mean(self.stats[prompt])
                std = np.std(self.stats[prompt]) + 1e-6
            advantages[prompts == prompt] = (prompt_rewards - mean) / std

        return advantages

    def get_stats(self):
        return {
            k: {"mean": np.mean(v), "std": np.std(v), "count": len(v)}
            for k, v in self.stats.items()
        }
