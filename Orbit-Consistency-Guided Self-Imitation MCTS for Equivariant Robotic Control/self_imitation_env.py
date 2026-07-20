import numpy as np
import torch
import math

class SelfImitationNavigationEnv:
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
        
        self.start = (1, 1)
        self.goal = (size - 2, size - 2)
        
        # Static obstacles coordinates (canonical setup for size=13)
        if size == 13:
            self.obstacles = {
                (3, 3), (3, 4), (3, 5),
                (5, 7), (6, 7), (7, 7),
                (9, 2), (9, 3), (9, 4),
                (8, 9), (9, 9), (10, 9)
            }
        else:
            self.obstacles = set()
            
        self._canonical_obstacles = frozenset(self.obstacles)

    def reset_canonical_obstacles(self):
        """Restores the original static obstacle layout."""
        self.obstacles = set(self._canonical_obstacles)

    def randomize_obstacles(self, n_obstacles=12, rng=None):
        """
        Randomly places n_obstacles on the grid while guaranteeing a valid
        BFS path exists from start to goal. Safe zones (2-cell radius around
        start and goal) are never blocked.
        """
        if rng is None:
            rng = np.random
        safe = set()
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                r, c = self.start[0] + dr, self.start[1] + dc
                if 0 <= r < self.size and 0 <= c < self.size:
                    safe.add((r, c))
                r, c = self.goal[0] + dr, self.goal[1] + dc
                if 0 <= r < self.size and 0 <= c < self.size:
                    safe.add((r, c))

        candidates = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) not in safe
        ]

        for _ in range(200):          # up to 200 retries
            chosen = rng.choice(len(candidates), size=min(n_obstacles, len(candidates)),
                                replace=False)
            obs = {candidates[i] for i in chosen}
            if self._bfs_path_exists(obs):
                self.obstacles = obs
                return
        # Fallback: use canonical obstacles if random placement keeps failing
        self.obstacles = set(self._canonical_obstacles)

    def _bfs_path_exists(self, obstacles):
        """Returns True if a valid 8-directional path exists from start to goal."""
        from collections import deque
        visited = set()
        queue = deque([self.start])
        visited.add(self.start)
        while queue:
            r, c = queue.popleft()
            if (r, c) == self.goal:
                return True
            for dr, dc in self.action_vectors:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.size and 0 <= nc < self.size
                        and (nr, nc) not in obstacles
                        and (nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return False

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
        if (r, c) in self.obstacles:
            return True, 2

        # Check if agent is missing (implies walked out of bounds)
        found_agent = False
        for row in state:
            if 1 in row:
                found_agent = True
                break
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
