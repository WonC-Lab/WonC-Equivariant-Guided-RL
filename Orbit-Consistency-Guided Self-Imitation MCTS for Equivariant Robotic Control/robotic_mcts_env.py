import numpy as np
import torch
import copy

class RoboticMCTSEnv:
    """
    State-based 3D Kinematic Robot Arm Environment Wrapper for MCTS.
    Discretizes the Cartesian velocity control space to 7 discrete actions:
      0: Stop
      1: +X direction
      2: -X direction
      3: +Y direction
      4: -Y direction
      5: +Z direction
      6: -Z direction
    """
    def __init__(self, l1=1.0, l2=1.0, l3=1.0, dt=0.05, max_steps=100):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.dt = dt
        self.max_steps = max_steps
        
        # Joint limits
        self.joint_min = np.array([-np.pi, -np.pi/2, -np.pi/2])
        self.joint_max = np.array([np.pi, np.pi/2, np.pi/2])
        
        # Discretized actions
        s = 0.5  # velocity step size
        self.action_vectors = [
            np.array([0.0, 0.0, 0.0]),  # 0: Stop
            np.array([s, 0.0, 0.0]),    # 1: +X
            np.array([-s, 0.0, 0.0]),   # 2: -X
            np.array([0.0, s, 0.0]),    # 3: +Y
            np.array([0.0, -s, 0.0]),   # 4: -Y
            np.array([0.0, 0.0, s]),    # 5: +Z
            np.array([0.0, 0.0, -s])    # 6: -Z
        ]
        
        self.obstacle_radius = 0.12 # 12cm radius obstacle

    def randomize_obstacles(self):
        """Randomly generates reachable target and places an obstacle between start and target."""
        # Generate target
        r = np.random.uniform(0.5, self.l2 + self.l3)
        yaw = np.random.uniform(-np.pi/2, np.pi/2)  # forward sector
        pitch = np.random.uniform(-np.pi/4, np.pi/4)
        
        x = r * np.cos(pitch) * np.cos(yaw)
        y = r * np.cos(pitch) * np.sin(yaw)
        z = self.l1 + r * np.sin(pitch)
        target = np.array([x, y, z], dtype=np.float32)
        
        # Set start joints
        theta = np.array([0.0, np.pi/6, -np.pi/6]) + np.random.uniform(-0.05, 0.05, size=3)
        theta = np.clip(theta, self.joint_min, self.joint_max)
        ee_start = self.forward_kinematics(theta)
        
        # Place obstacle midway
        alpha = np.random.uniform(0.4, 0.6)
        noise_obs = np.random.uniform(-0.05, 0.05, size=3)
        obstacle = alpha * ee_start + (1.0 - alpha) * target + noise_obs
        
        return {
            "theta": theta.tolist(),
            "target": target.tolist(),
            "obstacle": obstacle.tolist(),
            "steps": 0
        }

    def forward_kinematics(self, theta):
        th1, th2, th3 = theta[0], theta[1], theta[2]
        c1, s1 = np.cos(th1), np.sin(th1)
        c2, s2 = np.cos(th2), np.sin(th2)
        c23, s23 = np.cos(th2 + th3), np.sin(th2 + th3)
        
        x = c1 * (self.l2 * c2 + self.l3 * c23)
        y = s1 * (self.l2 * c2 + self.l3 * c23)
        z = self.l1 - (self.l2 * s2 + self.l3 * s23)
        return np.array([x, y, z], dtype=np.float32)

    def get_jacobian(self, theta):
        th1, th2, th3 = theta[0], theta[1], theta[2]
        c1, s1 = np.cos(th1), np.sin(th1)
        c2, s2 = np.cos(th2), np.sin(th2)
        c23, s23 = np.cos(th2 + th3), np.sin(th2 + th3)
        
        J = np.zeros((3, 3), dtype=np.float32)
        J[0, 0] = -s1 * (self.l2 * c2 + self.l3 * c23)
        J[1, 0] = c1 * (self.l2 * c2 + self.l3 * c23)
        J[2, 0] = 0.0
        
        J[0, 1] = c1 * (-self.l2 * s2 - self.l3 * s23)
        J[1, 1] = s1 * (-self.l2 * s2 - self.l3 * s23)
        J[2, 1] = -self.l2 * c2 - self.l3 * c23
        
        J[0, 2] = c1 * (-self.l3 * s23)
        J[1, 2] = s1 * (-self.l3 * s23)
        J[2, 2] = -self.l3 * c23
        return J

    def clone_state(self, state):
        return copy.deepcopy(state)

    def step(self, state, action, turn=1):
        next_state = self.clone_state(state)
        theta = np.array(next_state["theta"])
        
        # Cartesian action vector
        vel_cmd = self.action_vectors[action]
        
        # Damped Least Squares Jacobian pseudo-inverse
        J = self.get_jacobian(theta)
        damping = 0.01
        JJ_T = np.dot(J, J.T)
        inv_term = np.linalg.inv(JJ_T + (damping ** 2) * np.eye(3))
        J_pinv = np.dot(J.T, inv_term)
        
        dtheta = np.dot(J_pinv, vel_cmd)
        
        # Integrate and clip joint positions
        theta_next = theta + dtheta * self.dt
        theta_next = np.clip(theta_next, self.joint_min, self.joint_max)
        
        next_state["theta"] = theta_next.tolist()
        next_state["steps"] += 1
        
        return next_state, 1

    def check_game_over(self, state, turn=1):
        """
        Checks terminal state conditions.
        Returns: (is_over, winner)
        - winner = 1: Reached target (Success)
        - winner = 2: Collided with obstacle / Exceeded steps (Failure)
        """
        theta = np.array(state["theta"])
        target = np.array(state["target"])
        obstacle = np.array(state["obstacle"])
        
        ee_pos = self.forward_kinematics(theta)
        
        dist = np.linalg.norm(ee_pos - target)
        if dist < 0.25:
            return True, 1
            
        obs_dist = np.linalg.norm(ee_pos - obstacle)
        if obs_dist < self.obstacle_radius:
            return True, 2
            
        if state["steps"] >= self.max_steps:
            return True, 2
            
        return False, None

    def get_valid_actions(self, state, turn=1):
        # All 7 actions are kinematic commands and valid
        return list(range(7))

    def state_to_tensor(self, state, turn=1):
        """
        Converts Cartesian state locations to a (1, 3, 3) spatial Tensor for SO(3) frame-averaging.
        Row 0: End-effector pos [x, y, z]
        Row 1: Target pos [x, y, z]
        Row 2: Obstacle pos [x, y, z]
        """
        theta = np.array(state["theta"])
        ee_pos = self.forward_kinematics(theta)
        target = np.array(state["target"])
        obstacle = np.array(state["obstacle"])
        
        # Stack coordinates: (3, 3)
        coords = np.stack([ee_pos, target, obstacle], axis=0)
        return torch.tensor(coords, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 3, 3)
