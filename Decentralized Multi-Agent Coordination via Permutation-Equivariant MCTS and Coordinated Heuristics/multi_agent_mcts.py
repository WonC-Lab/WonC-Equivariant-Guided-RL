import math
import numpy as np
import torch

class MCTSNode:
    """
    Represents a single node in the MCTS search tree for a specific agent.
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
        action_probs: Dictionary mapping action index (0-7) -> probability.
        """
        for action, prob in action_probs.items():
            if action not in self.children:
                self.children[action] = MCTSNode(parent=self, prior_prob=prob)

    def is_expanded(self):
        return len(self.children) > 0


class MultiAgentMCTS:
    """
    Decentralized Actor-Critic MCTS for Multi-Agent Reinforcement Learning.
    Each agent runs an independent MCTS tree.
    During search traversal, other agents are modeled as static obstacles to preserve
    decentralization and computational efficiency, eliminating multi-agent network queries during Selection.
    """
    def __init__(self, model, c_puct=1.5):
        self.model = model
        self.c_puct = c_puct

    def get_puct_value(self, node, child):
        """
        Calculates PUCT: Q(s, a) + U(s, a)
        """
        u = (self.c_puct * child.prior_prob * 
             math.sqrt(node.visit_count) / (1 + child.visit_count))
        return child.value + u

    def run_simulation(self, root_state, agent_idx, env, beta=0.0):
        """
        Runs a single MCTS iteration for agent_idx.
        """
        original_obstacles = env.obstacles.copy()
        agent_positions, _, _, root_active_mask = root_state
        for j in range(env.num_agents):
            if j != agent_idx and root_active_mask[j]:
                env.obstacles.add(agent_positions[j])
                
        try:
            node = self.root
            state = root_state
            
            # Start with a mask where only agent_idx is active (if it was active at root)
            sim_active_mask = list(root_state[3])
            for j in range(env.num_agents):
                if j != agent_idx:
                    sim_active_mask[j] = False
            
            # 1. Selection phase
            while node.is_expanded():
                if not sim_active_mask[agent_idx]:
                    break
                    
                # Choose the action for agent_idx maximizing PUCT
                a_self, next_node = max(
                    node.children.items(),
                    key=lambda item: self.get_puct_value(node, item[1])
                )
                
                agent_positions, goal_positions, obstacles_positions, _ = state
                
                curr_state = (agent_positions, goal_positions, obstacles_positions, tuple(sim_active_mask))
                joint_action = [0] * env.num_agents
                joint_action[agent_idx] = a_self
                
                # Step in environment (other agents remain static)
                next_state, _, _, _ = env.step(curr_state, tuple(joint_action))
                
                # Update running simulation active mask
                sim_active_mask = list(next_state[3])
                
                # Keep state updated
                state = (next_state[0], next_state[1], next_state[2], tuple(sim_active_mask))
                node = next_node

            # 2. Expansion & Evaluation phase
            agent_positions, goal_positions, _, active_mask = state
            agent_done = not active_mask[agent_idx]
            
            if not agent_done:
                # Evaluate using GNN (Only 1 forward pass at the leaf node, following AlphaZero)
                self.model.eval()
                
                # GNN observation should see other agents as active at their current positions
                gnn_active_mask = list(root_state[3])
                # Ensure agent_idx's status in GNN state is updated
                gnn_active_mask[agent_idx] = active_mask[agent_idx]
                gnn_state = (state[0], state[1], state[2], tuple(gnn_active_mask))
                
                obs_tensor = env.get_joint_observation(gnn_state).unsqueeze(0)
                with torch.no_grad():
                    logits, values = self.model(obs_tensor)
                    
                value = min(0.95, values[0, agent_idx].item())
                
                # Mask logits for agent_idx
                valid_actions = env.get_valid_actions(state, agent_idx)
                a_logits = logits[0, agent_idx].clone()
                invalid_actions = [a for a in range(8) if a not in valid_actions]
                a_logits[invalid_actions] = -1e9
                
                gnn_probs = torch.softmax(a_logits, dim=0).cpu().numpy()
                
                if beta > 0.0:
                    heur_probs = env.get_heuristic_policy(state, agent_idx)
                    probs = (1.0 - beta) * gnn_probs + beta * heur_probs
                    # Normalize probs over valid actions
                    sum_p = sum([probs[act] for act in valid_actions])
                    if sum_p > 0:
                        probs = probs / sum_p
                else:
                    probs = gnn_probs
                    
                masked_probs = {act: probs[act] for act in valid_actions}
                
                # Expand the node
                node.expand(masked_probs)
            else:
                # Terminal evaluation for this agent
                pos = agent_positions[agent_idx]
                goal = goal_positions[agent_idx]
                if pos == goal:
                    value = 1.0
                else:
                    value = -1.0 # Crashed

            # 3. Backpropagation
            v = value
            while node is not None:
                node.visit_count += 1
                node.value_sum += v
                node = node.parent
                
        finally:
            env.obstacles = original_obstacles

    def get_action_probabilities(self, state, agent_idx, env, num_searches=100, temp=1.0, beta=0.0):
        """
        Performs multiple simulations to estimate the best action probability distribution.
        """
        self.root = MCTSNode()
        
        # Run simulations
        for _ in range(num_searches):
            self.run_simulation(state, agent_idx, env, beta=beta)
            
        if not self.root.children:
            return list(range(8)), np.ones(8) / 8.0

        # Extract visit counts
        actions = list(self.root.children.keys())
        visit_counts = [child.visit_count for child in self.root.children.values()]
        
        if temp == 0.0:
            best_idx = np.argmax(visit_counts)
            probs = np.zeros(len(actions))
            probs[best_idx] = 1.0
        else:
            counts_temp = np.array(visit_counts) ** (1.0 / temp)
            probs = counts_temp / np.sum(counts_temp)
            
        # Re-align probs to map to action indices 0..7
        full_probs = np.zeros(8)
        for act, prob in zip(actions, probs):
            full_probs[act] = prob
            
        return list(range(8)), full_probs
