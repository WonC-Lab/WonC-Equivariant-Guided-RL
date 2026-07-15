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
            # Move tensor to the same device as the model
            device = next(self.model.parameters()).device
            state_tensor = state_tensor.to(device)
            with torch.no_grad():
                policy_logits, value_tensor = self.model(state_tensor)
                
            value = value_tensor.item()
            
            # Mask invalid actions to -1e9 before softmax to prevent out-of-bounds logits from dominating
            valid_actions = game_env.get_valid_actions(state, turn)
            masked_logits = policy_logits.squeeze(0).clone()
            invalid_actions = [i for i in range(8) if i not in valid_actions]
            masked_logits[invalid_actions] = -1e9
            
            probs = torch.softmax(masked_logits, dim=0).cpu().numpy()
            masked_probs = {act: probs[act] for act in valid_actions}

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
        # Note: Values do not alternate signs for single-agent MDP
        v = value
        while node is not None:
            node.visit_count += 1
            node.value_sum += v
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
            best_idx = np.argmax(visit_counts)
            probs = np.zeros(len(actions))
            probs[best_idx] = 1.0
        else:
            counts_temp = np.array(visit_counts) ** (1.0 / temp)
            probs = counts_temp / np.sum(counts_temp)

        return actions, probs
