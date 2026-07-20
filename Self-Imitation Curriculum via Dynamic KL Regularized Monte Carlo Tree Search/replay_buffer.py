import random
import numpy as np
import torch
from models import D4GroupAction

class SymmetricSelfImitationBuffer:
    """
    Symmetric Prioritized Replay Buffer.
    Extends every stored transition with 8 symmetric versions using D_4 Dihedral transformations,
    greatly boosting sample efficiency in symmetric environments.
    """
    def __init__(self, max_size=20000, running_avg_window=50):
        self.max_size = max_size
        self.buffer = []  # List of tuples: (state_grid_list, mcts_prob_np, return_g)
        self.episode_returns = []
        self.running_avg_window = running_avg_window
        self.best_episode_returns = []

    def get_running_average(self):
        if not self.episode_returns:
            return -10.0  # low baseline to allow initial uploads
        window = min(len(self.episode_returns), self.running_avg_window)
        return np.mean(self.episode_returns[-window:])

    def get_mean_best_reward(self):
        if not self.best_episode_returns:
            return 0.0
        return np.mean(self.best_episode_returns)

    def add_episode(self, episode_transitions, total_return):
        """
        Adds transitions to the buffer. If the episode is added, generates
        8 symmetric D_4 variants for each transition.
        """
        running_avg = self.get_running_average()
        self.episode_returns.append(total_return)

        # Buffer criteria: better than running average or any successful run (total_return > 0)
        if len(self.episode_returns) <= 10 or total_return > running_avg or total_return > 0:
            self.best_episode_returns.append(total_return)
            
            # Apply D_4 Group Expansion to each transition
            for state, mcts_prob, return_g in episode_transitions:
                # 1. State tensor transform setup
                # State list is shape (13, 13)
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 13, 13)
                
                for i in range(8):
                    # Apply group action to grid state
                    g_state_t = D4GroupAction.apply_action(state_t, i)
                    g_state = g_state_t.squeeze(0).squeeze(0).numpy().astype(np.int32).tolist()
                    
                    # Apply action permutation to the probability distribution
                    # P_curriculum(g*a | g*s) = P_curriculum(a | s) => g_probs[perm_i[a]] = probs[a]
                    perm_i = D4GroupAction.get_action_permutation(i)
                    g_prob = np.zeros(8)
                    for act_idx, p in enumerate(mcts_prob):
                        g_prob[perm_i[act_idx]] = p
                        
                    self.buffer.append((g_state, g_prob, return_g))

            # Prune if over maximum size
            if len(self.buffer) > self.max_size:
                self.buffer = self.buffer[-self.max_size:]

            if len(self.best_episode_returns) > 100:
                self.best_episode_returns = self.best_episode_returns[-100:]
            return True
        return False

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            samples = random.choices(self.buffer, k=batch_size)
        else:
            samples = random.sample(self.buffer, batch_size)
        states, mcts_probs, returns = zip(*samples)
        return states, mcts_probs, returns

    def __len__(self):
        return len(self.buffer)
