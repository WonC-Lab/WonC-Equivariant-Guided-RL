import numpy as np
import torch
import torch.optim as optim
from autonomous_env import AutonomousNavigationEnv
from equivariant_models import D4EquivariantNet
from mcts_actor_critic import ActorCriticMCTS
from heuristic_guided_loss import HeuristicGuidedLoss

class SymmetricNavEnvAdapter:
    """
    Adapts the 8-directional AutonomousNavigationEnv to a direct 8-directional
    action space to natively interoperate with the updated D4EquivariantNet and MCTS.
    """
    def __init__(self, base_env):
        self.base_env = base_env
        self.size = base_env.size

    def clone_state(self, state):
        return self.base_env.clone_state(state)

    def step(self, state, action, turn=1):
        return self.base_env.step(state, action, turn)

    def check_game_over(self, state, turn=1):
        return self.base_env.check_game_over(state, turn)

    def get_valid_actions(self, state, turn=1):
        return self.base_env.get_valid_actions(state, turn)

    def state_to_tensor(self, state, turn=1):
        return self.base_env.state_to_tensor(state, turn)

    def get_heuristic_policy_flat(self, state):
        return self.base_env.get_heuristic_policy(state)


def train_agent():
    print("=============================================================")
    print(" Starting Symmetric & Heuristic-Guided Navigation Training...")
    print("=============================================================")

    base_env = AutonomousNavigationEnv(size=13)
    env = SymmetricNavEnvAdapter(base_env)
    
    model = D4EquivariantNet(board_size=13, in_channels=3, num_filters=32, num_layers=6)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    mcts = ActorCriticMCTS(model=model, c_puct=1.4)
    loss_fn = HeuristicGuidedLoss(beta_start=1.0, beta_decay=0.96, beta_min=0.5)

    num_episodes = 25
    max_steps_per_episode = 25
    gamma = 0.95

    success_count = 0

    for ep in range(1, num_episodes + 1):
        state = base_env.generate_initial_state()
        
        episode_states = []
        episode_actions = []
        episode_heuristics = []
        episode_old_log_probs = []
        
        step = 0
        game_over = False
        winner = None

        while not game_over and step < max_steps_per_episode:
            state_tensor = env.state_to_tensor(state, turn=1)
            
            # Get old log probability under current policy
            model.eval()
            with torch.no_grad():
                logits, _ = model(state_tensor)
            log_probs = torch.log_softmax(logits, dim=-1)

            actions, mcts_probs = mcts.get_action_probabilities(
                state, current_turn=1, game_env=env, num_searches=30, temp=1.0
            )
            
            search_policy_probs = np.zeros(13 * 13, dtype=np.float32)
            for act, prob in zip(actions, mcts_probs):
                search_policy_probs[act] = prob

            heuristic_probs = env.get_heuristic_policy_flat(state)
            
            chosen_action = np.random.choice(actions, p=mcts_probs)
            old_log_prob = log_probs[0, chosen_action].item()
            
            episode_states.append(state_tensor)
            episode_actions.append(chosen_action)
            episode_heuristics.append(heuristic_probs)
            episode_old_log_probs.append(old_log_prob)

            state, _ = env.step(state, chosen_action, turn=1)
            game_over, winner = env.check_game_over(state, turn=1)
            step += 1

        if winner == 1:
            final_reward = 1.0
            success_count += 1
            result_str = "SUCCESS (Goal Reached)"
        elif winner == 2:
            final_reward = -1.0
            result_str = "CRASHED (Obstacle/Out of bounds)"
        else:
            final_reward = -0.2
            result_str = "TIMEOUT (Max steps exceeded)"

        returns = []
        g = final_reward
        for _ in reversed(episode_states):
            returns.append(g)
            g *= gamma
        returns.reverse()

        if episode_states:
            model.train()

            batch_states = torch.cat(episode_states, dim=0)
            batch_actions = torch.tensor(episode_actions, dtype=torch.long)
            batch_returns = torch.tensor(returns, dtype=torch.float32)
            batch_heuristics = torch.tensor(np.array(episode_heuristics), dtype=torch.float32)
            batch_old_log_probs = torch.tensor(episode_old_log_probs, dtype=torch.float32)

            avg_loss = 0.0
            for _ in range(5):
                optimizer.zero_grad()
                logits, values = model(batch_states)
                loss, rl_loss_val, kl_loss_val, val_loss_val = loss_fn(
                    logits, values, batch_actions, batch_returns, batch_heuristics, batch_old_log_probs
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                avg_loss += loss.item()
            avg_loss /= 5.0
            loss_fn.decay_beta()

            print(f"Episode {ep:02d}/{num_episodes} | Steps: {step:02d} | Result: {result_str:<32} | Loss: {avg_loss:.4f} | Beta: {loss_fn.beta:.4f}")
        else:
            print(f"Episode {ep:02d}/{num_episodes} | Steps: {step:02d} | Result: Empty Episode")

    print("=============================================================")
    print(f" Training Completed. Success Rate: {success_count / num_episodes * 100:.1f}%")
    print("=============================================================")


if __name__ == "__main__":
    train_agent()
