import numpy as np
import torch

class AutonomousNavigationEnv:
    """
    Symmetric 13x13 Grid Map Navigation Environment.
    - Start point: (1, 1)
    - Goal point: (11, 11)
    - Obstacles: Distributed statically across the grid.
    - Actions: 8-directional movement.
    """
    def __init__(self, size=13):
        self.size = size
        # Define 8 moves: Up, Down, Left, Right, Up-Left, Up-Right, Down-Left, Down-Right
        self.action_vectors = [
            (-1, 0),  # 0: Up
            (1, 0),   # 1: Down
            (0, -1),  # 2: Left
            (0, 1),   # 3: Right
            (-1, -1), # 4: Up-Left
            (-1, 1),  # 5: Up-Right
            (1, -1),  # 6: Down-Left
            (1, 1)    # 7: Down-Right
        ]
        
        # Static obstacles coordinates
        self.obstacles = {
            (3, 3), (3, 4), (3, 5),
            (5, 7), (6, 7), (7, 7),
            (9, 2), (9, 3), (9, 4),
            (8, 9), (9, 9), (10, 9)
        }
        self.start = (1, 1)
        self.goal = (11, 11)

    def generate_initial_state(self):
        """
        Generates clean initial state grid.
        - Agent = 1, Goal = 2, Obstacle = 3, Empty = 0
        """
        grid = np.zeros((self.size, self.size), dtype=np.int32)
        for r, c in self.obstacles:
            grid[r, c] = 3
        grid[self.goal[0], self.goal[1]] = 2
        grid[self.start[0], self.start[1]] = 1
        return grid.tolist()

    def get_agent_pos(self, state):
        """Finds agent's coordinate (r, c) on the grid."""
        for r in range(self.size):
            for c in range(self.size):
                if state[r][c] == 1:
                    return r, c
        return self.start # fallback

    def clone_state(self, state):
        return [row[:] for row in state]

    def in_bounds(self, r, c):
        return 0 <= r < self.size and 0 <= c < self.size

    def step(self, state, action, turn=1):
        """
        Executes agent movement.
        MCTS interface expects (state, action, turn) and returns (next_state, next_turn).
        For single-agent navigation, we keep turn = 1.
        """
        next_state = self.clone_state(state)
        r, c = self.get_agent_pos(next_state)
        
        dr, dc = self.action_vectors[action]
        nr, nc = r + dr, c + dc

        # Remove old agent position
        next_state[r][c] = 0
        # Re-draw Goal or Obstacles if agent left that spot
        if (r, c) == self.goal:
            next_state[r][c] = 2
        elif (r, c) in self.obstacles:
            next_state[r][c] = 3

        # Place agent at new position if in bounds
        if self.in_bounds(nr, nc):
            next_state[nr][nc] = 1
            
        return next_state, 1

    def check_game_over(self, state, turn=1):
        """
        Checks terminal states.
        Returns: (is_over, winner)
        - winner = 1: Reached Goal (Success)
        - winner = 2: Collided with Obstacles / Out of Bounds (Failure)
        """
        r, c = self.get_agent_pos(state)
        
        # Success check
        if (r, c) == self.goal:
            return True, 1

        # Collision check (if agent position overlaps with obstacle)
        # Note: If agent walked out of bounds, nr/nc wasn't placed, keeping agent pos unchanged or gone.
        # We also check if the agent coordinate matches any obstacle.
        if (r, c) in self.obstacles:
            return True, 2

        # Check if agent is missing (implies walked out of bounds)
        # If no 1 is present in grid, game over as crash.
        found_agent = False
        for row in state:
            if 1 in row:
                found_agent = True; break
        if not found_agent:
            return True, 2

        return False, None

    def get_valid_actions(self, state, turn=1):
        """Returns valid action indices that keep agent inside the grid."""
        r, c = self.get_agent_pos(state)
        valid = []
        for i, (dr, dc) in enumerate(self.action_vectors):
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc):
                valid.append(i)
        return valid

    def state_to_tensor(self, state, turn=1):
        """
        Translates state grid into (1, 3, size, size) PyTorch Tensor.
        - Channel 0: Agent position
        - Channel 1: Obstacles positions
        - Channel 2: Goal position
        """
        board = np.array(state)
        ch0 = (board == 1).astype(np.float32)
        ch1 = (board == 3).astype(np.float32)
        ch2 = (board == 2).astype(np.float32)
        return torch.tensor(np.stack([ch0, ch1, ch2], axis=0)).unsqueeze(0)

    def get_heuristic_policy(self, state):
        """
        Generates a heuristic target policy (probability distribution over 8 actions).
        Heuristic: Prefers moves that reduce Euclidean distance to the Goal without hitting obstacles.
        """
        r, c = self.get_agent_pos(state)
        gr, gc = self.goal
        
        scores = []
        for i, (dr, dc) in enumerate(self.action_vectors):
            nr, nc = r + dr, c + dc
            if not self.in_bounds(nr, nc) or (nr, nc) in self.obstacles:
                # Assign extremely low score to collisions/out-of-bounds
                scores.append(-9999.0)
            else:
                # Calculate negative distance (closer is better)
                dist = math.sqrt((nr - gr)**2 + (nc - gc)**2)
                scores.append(-dist)
                
        # Apply Softmax to convert scores to probability distribution
        scores = np.array(scores)
        exp_s = np.exp(scores - np.max(scores))
        probs = exp_s / np.sum(exp_s)
        return probs

import math
