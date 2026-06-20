import math
import numpy as np
import torch

class MCTSNode:
    """
    Represents a single node in the MCTS search tree.
    """
    def __init__(self, parent=None, prior_prob=1.0):
        self.parent = parent
        self.children = {}  # Map: action -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_prob = prior_prob

    @property
    def value(self):
        """Returns the mean action value Q(s, a)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, action_probs):
        """
        Expands the leaf node by creating children with prior probabilities from the policy net.
        action_probs: Dictionary or list mapping action -> probability.
        """
        for action, prob in action_probs.items():
            if action not in self.children:
                self.children[action] = MCTSNode(parent=self, prior_prob=prob)

    def is_expanded(self):
        return len(self.children) > 0


class ActorCriticMCTS:
    """
    Implements AlphaZero-style Monte Carlo Tree Search using a neural network evaluator.
    """
    def __init__(self, model, c_puct=1.5):
        self.model = model
        self.c_puct = c_puct

    def get_puct_value(self, node, child):
        """
        Calculates PUCT formula: Q(s, a) + U(s, a)
        """
        u = (self.c_puct * child.prior_prob * 
             math.sqrt(node.visit_count) / (1 + child.visit_count))
        return child.value + u

    def run_simulation(self, root_state, current_turn, game_env):
        """
        Executes a single MCTS iteration (Selection -> Expansion -> Evaluation -> Backpropagation).
        """
        node = self.root
        state = game_env.clone_state(root_state)
        turn = current_turn

        # 1. Selection: Traverse down using PUCT until a leaf node is hit
        while node.is_expanded():
            action, node = max(
                node.children.items(),
                key=lambda item: self.get_puct_value(node, item[1])
            )
            state, turn = game_env.step(state, action, turn)

        # Check if the game has ended at this node
        game_over, winner = game_env.check_game_over(state, turn)
        
        if not game_over:
            # 2. Evaluation: Query the Actor-Critic model for policy prior and state value
            self.model.eval()
            state_tensor = game_env.state_to_tensor(state, turn)
            with torch.no_grad():
                policy_logits, value_tensor = self.model(state_tensor)
                
            value = value_tensor.item()
            probs = torch.softmax(policy_logits.squeeze(0), dim=0).cpu().numpy()

            # Filter valid actions (Mask illegal moves)
            valid_actions = game_env.get_valid_actions(state, turn)
            masked_probs = {act: probs[act] for act in valid_actions}
            
            # Normalize probabilities
            total_prob = sum(masked_probs.values())
            if total_prob > 0:
                masked_probs = {k: v / total_prob for k, v in masked_probs.items()}
            else:
                # Uniform fallback if neural net output is zeroed out
                masked_probs = {k: 1.0 / len(valid_actions) for k in valid_actions}

            # 3. Expansion: Expand node with filtered actions
            node.expand(masked_probs)
        else:
            # Game is over, determine actual terminal value
            if winner == current_turn:
                value = 1.0
            elif winner == (3 - current_turn):
                value = -1.0
            else:
                value = 0.0

        # 4. Backpropagation: Update tree metrics up to the root
        # Note: Values alternate signs depending on the active player perspective
        v = value
        while node is not None:
            node.visit_count += 1
            node.value_sum += v
            v = -v  # Flip value perspective for parent node (Minimax view)
            node = node.parent

    def get_action_probabilities(self, state, current_turn, game_env, num_searches=100, temp=1.0):
        """
        Performs multiple tree searches and returns the policy distribution based on visit counts.
        """
        self.root = MCTSNode()

        # Run simulations
        for _ in range(num_searches):
            self.run_simulation(state, current_turn, game_env)

        # Extract visit counts from root's children
        actions = list(self.root.children.keys())
        visit_counts = [child.visit_count for child in self.root.children.values()]

        if temp == 0:
            # Deterministic: choose the action with the maximum visit count
            best_idx = np.argmax(visit_counts)
            probs = np.zeros(len(actions))
            probs[best_idx] = 1.0
        else:
            # Softmax distribution over visit counts based on temperature
            counts_temp = np.array(visit_counts) ** (1.0 / temp)
            probs = counts_temp / np.sum(counts_temp)

        return actions, probs


# Mock Environment Interface to allow verification compile
class MockGameEnvironment:
    """
    An abstract template environment showing how game frameworks should interface with MCTS.
    """
    def __init__(self, size=13):
        self.size = size

    def clone_state(self, state):
        return [row[:] for row in state]

    def step(self, state, action, turn):
        r, c = action // self.size, action % self.size
        state[r][c] = turn
        return state, 3 - turn

    def check_game_over(self, state, turn):
        # returns (is_over, winner)
        return False, None

    def get_valid_actions(self, state, turn):
        flat_state = np.array(state).flatten()
        return np.where(flat_state == 0)[0]

    def state_to_tensor(self, state, turn):
        # Converts board state to (1, 3, size, size) tensor
        board = np.array(state)
        ch0 = (board == turn).astype(np.float32)
        ch1 = (board == (3 - turn)).astype(np.float32)
        ch2 = (board == 0).astype(np.float32)
        return torch.tensor(np.stack([ch0, ch1, ch2], axis=0)).unsqueeze(0)


if __name__ == "__main__":
    from equivariant_models import D4EquivariantNet

    print("Verifying Actor-Critic MCTS execution pipeline...")
    
    # 1. Initialize models and environment
    model = D4EquivariantNet(board_size=13, in_channels=3, num_filters=16, num_layers=2)
    env = MockGameEnvironment(size=13)
    mcts = ActorCriticMCTS(model=model, c_puct=1.5)

    # 2. Setup an empty board state
    initial_board = [[0] * 13 for _ in range(13)]
    current_turn = 1

    # 3. Request action probability distributions
    actions, probs = mcts.get_action_probabilities(
        initial_board, current_turn, env, num_searches=50, temp=1.0
    )

    print("MCTS Search succeeded!")
    print(f"Top 5 action candidates: {actions[:5]}")
    print(f"Corresponding visit probabilities: {probs[:5]}")
    assert len(actions) == len(probs), "Actions and probabilities size mismatch!"
    print("OK: MCTS execution verified successfully.")
