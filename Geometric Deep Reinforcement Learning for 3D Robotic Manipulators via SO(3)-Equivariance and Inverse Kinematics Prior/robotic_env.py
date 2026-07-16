import numpy as np
import torch

class RoboticArm3DEnv:
    """
    A lightweight, pure-Python 3D kinematic robot arm environment (3-DOF).
    The joints consist of:
      1. Base rotation (revolute, around Z-axis): theta_1
      2. Shoulder rotation (revolute, around Y-axis): theta_2
      3. Elbow rotation (revolute, around Y-axis): theta_3
      
    The state is defined by the end-effector coordinate and a target coordinate.
    The action is the Cartesian end-effector velocity v_ee in R^3.
    We convert the action to joint velocities via the Damped Least Squares (DLS) Jacobian pseudoinverse.
    """
    def __init__(self, l1=1.0, l2=1.0, l3=1.0, dt=0.05, max_steps=200):
        self.l1 = l1  # Base height / link 1 length
        self.l2 = l2  # Link 2 length
        self.l3 = l3  # Link 3 length
        self.dt = dt
        self.max_steps = max_steps
        
        # Joint limits
        self.joint_min = np.array([-np.pi, -np.pi/2, -np.pi/2])
        self.joint_max = np.array([np.pi, np.pi/2, np.pi/2])
        
        self.reset()
        
    def reset(self, target_pos=None, restrict_sector=False):
        # Reset joint angles to a default home position with some small noise
        self.theta = np.array([0.0, np.pi/6, -np.pi/6]) + np.random.uniform(-0.1, 0.1, size=3)
        self.theta = np.clip(self.theta, self.joint_min, self.joint_max)
        
        # Generate a target position
        if target_pos is not None:
            self.target = np.array(target_pos, dtype=np.float32)
        else:
            # Generate a reachable target in the workspace
            # Max reach is l2 + l3 from base z-height (l1)
            r = np.random.uniform(0.5, self.l2 + self.l3)
            if restrict_sector:
                # Restrict to x > 0: yaw in [-pi/2, pi/2]
                yaw = np.random.uniform(-np.pi/2, np.pi/2)
            else:
                yaw = np.random.uniform(-np.pi, np.pi)
            pitch = np.random.uniform(-np.pi/3, np.pi/3)
            
            x = r * np.cos(pitch) * np.cos(yaw)
            y = r * np.cos(pitch) * np.sin(yaw)
            z = self.l1 + r * np.sin(pitch)
            self.target = np.array([x, y, z], dtype=np.float32)
            
        # Get start position
        ee_start = self.forward_kinematics(self.theta)
        
        # Place obstacle between start and target with some random offset
        alpha = np.random.uniform(0.4, 0.6)
        noise_obs = np.random.uniform(-0.1, 0.1, size=3)
        self.obstacle = alpha * ee_start + (1.0 - alpha) * self.target + noise_obs
        self.obstacle_radius = 0.10 # 10cm radius spherical obstacle
        
        self.steps = 0
        return self._get_obs()

    def forward_kinematics(self, theta):
        """
        Computes forward kinematics for the 3-DOF robot arm.
        Returns the 3D position of the end-effector.
        """
        th1, th2, th3 = theta[0], theta[1], theta[2]
        
        c1, s1 = np.cos(th1), np.sin(th1)
        c2, s2 = np.cos(th2), np.sin(th2)
        c23, s23 = np.cos(th2 + th3), np.sin(th2 + th3)
        
        # End-effector coordinates
        x = c1 * (self.l2 * c2 + self.l3 * c23)
        y = s1 * (self.l2 * c2 + self.l3 * c23)
        z = self.l1 - (self.l2 * s2 + self.l3 * s23)
        
        return np.array([x, y, z], dtype=np.float32)

    def get_jacobian(self, theta):
        """
        Computes the analytical 3x3 geometric Jacobian matrix mapping
        joint angular velocities to end-effector Cartesian velocities:
        v_ee = J(theta) * d(theta)/dt
        """
        th1, th2, th3 = theta[0], theta[1], theta[2]
        
        c1, s1 = np.cos(th1), np.sin(th1)
        c2, s2 = np.cos(th2), np.sin(th2)
        c23, s23 = np.cos(th2 + th3), np.sin(th2 + th3)
        
        # J_11 = dx/dth1, J_12 = dx/dth2, J_13 = dx/dth3
        # J_21 = dy/dth1, J_22 = dy/dth2, J_23 = dy/dth3
        # J_31 = dz/dth1, J_32 = dz/dth2, J_33 = dz/dth3
        
        J = np.zeros((3, 3), dtype=np.float32)
        
        # d/dth1
        J[0, 0] = -s1 * (self.l2 * c2 + self.l3 * c23)
        J[1, 0] = c1 * (self.l2 * c2 + self.l3 * c23)
        J[2, 0] = 0.0
        
        # d/dth2
        J[0, 1] = c1 * (-self.l2 * s2 - self.l3 * s23)
        J[1, 1] = s1 * (-self.l2 * s2 - self.l3 * s23)
        J[2, 1] = -self.l2 * c2 - self.l3 * c23
        
        # d/dth3
        J[0, 2] = c1 * (-self.l3 * s23)
        J[1, 2] = s1 * (-self.l3 * s23)
        J[2, 2] = -self.l3 * c23
        
        return J

    def _get_obs(self):
        ee_pos = self.forward_kinematics(self.theta)
        # Return observation as a tensor of shape [3, 3] containing:
        # [ [ee_x, target_x, obs_x],
        #   [ee_y, target_y, obs_y],
        #   [ee_z, target_z, obs_z] ]
        obs = np.stack([ee_pos, self.target, self.obstacle], axis=1) # Shape: (3, 3)
        return obs.astype(np.float32)

    def step(self, action):
        """
        action: Cartesian end-effector velocity command in R^3
        """
        action = np.array(action, dtype=np.float32)
        
        # Get current Jacobian
        J = self.get_jacobian(self.theta)
        
        # Damped Least Squares (DLS) Jacobian pseudoinverse to handle singularities
        damping = 0.01
        JJ_T = np.dot(J, J.T)
        inv_term = np.linalg.inv(JJ_T + (damping ** 2) * np.eye(3))
        J_pinv = np.dot(J.T, inv_term)
        
        # Calculate joint velocities
        dtheta = np.dot(J_pinv, action)
        
        # Integrate joint positions
        self.theta = self.theta + dtheta * self.dt
        self.theta = np.clip(self.theta, self.joint_min, self.joint_max)
        
        # Get new observation and calculate reward
        obs = self._get_obs()
        ee_pos = obs[:, 0]
        
        dist = np.linalg.norm(ee_pos - self.target)
        
        # Obstacle collision check
        obs_dist = np.linalg.norm(ee_pos - self.obstacle)
        collision = obs_dist < self.obstacle_radius
        
        # Reward function
        reward = -dist - 0.05 * np.linalg.norm(action)**2
        
        if collision:
            reward -= 1.0 # Soft collision penalty per step (does not terminate)
            
        self.steps += 1
        done = (dist < 0.05) or (self.steps >= self.max_steps)
        
        info = {
            "dist": dist,
            "success": (dist < 0.05),
            "collision": collision,
            "ee_pos": ee_pos,
            "theta": self.theta
        }
        
        return obs, reward, done, info
