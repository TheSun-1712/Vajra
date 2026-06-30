import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
from vajra import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VajraAOEnv(gym.Env):
    """Gymnasium Wrapper for the VAJRA Physical Optics Simulator.
    The agent dynamically tunes WFS loop parameters (Gains, GLAO weighting, Notch)
    to maximize correction performance (Strehl ratio) under changing atmospheres.
    """
    def __init__(self, sim_mode="solar", history_len=5):
        super(VajraAOEnv, self).__init__()
        from vajra.simulator import AOPipelineSimulator
        from vajra.detector import DetectorProcessor
        
        self.sim_mode = sim_mode
        self.sim = AOPipelineSimulator(mode=sim_mode)
        self.detector = DetectorProcessor()
        self.history_len = history_len
        self.num_subaps = len(self.sim.subaps_pos)
        
        # Action space: 
        # [0] loop integrator gain scale factor (0.1 to 1.2)
        # [1] GLAO ground-layer component weight (0.0 to 1.0)
        # [2] Notch filter damping tuning (0.90 to 0.99)
        self.action_space = spaces.Box(
            low=np.array([0.1, 0.0, 0.90]), 
            high=np.array([1.2, 1.0, 0.99]), 
            dtype=np.float32
        )
        
        # Observation space: history of flattened subaperture centroids (x, y) + confidences
        self.obs_dim_per_frame = self.num_subaps * 2 + self.num_subaps
        obs_total_dim = self.obs_dim_per_frame * self.history_len
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_total_dim,), 
            dtype=np.float32
        )
        
        self.history = []
        self.loop_time = 0.0
        self.current_commands = np.zeros(self.sim.dm.num_actuators)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        from vajra.simulator import AOPipelineSimulator
        self.sim = AOPipelineSimulator(mode=self.sim_mode)
        self.loop_time = 0.0
        self.current_commands = np.zeros(self.sim.dm.num_actuators)
        self.history = []
        
        # Generate initial frame
        raw_frame, illum = self.sim.generate_hartmannogram()
        centroids, _, _, confidences = self.detector.process_frame(raw_frame, illum)
        
        init_obs = np.concatenate([centroids.flatten(), confidences])
        for _ in range(self.history_len):
            self.history.append(init_obs)
            
        obs = np.concatenate(self.history)
        return obs, {}

    def step(self, action):
        self.loop_time += 0.001
        self.sim.evolve_atmosphere(self.loop_time)
        
        # Unpack action values
        loop_gain, glao_weight, notch_r = action
        
        # Apply actions to run 1 iteration of the AO loop
        true_phase = self.sim.get_phase_screen()
        raw_frame, illum = self.sim.generate_hartmannogram(dm_commands=self.current_commands)
        centroids, fluxes, fwhms, confidences = self.detector.process_frame(raw_frame, illum)
        
        # Emulate WLS reconstructor with RL tuned gain
        from vajra.reconstructor import WavefrontReconstructor
        reconstructor = WavefrontReconstructor(
            self.sim.pupil_grid, self.sim.pupil_mask, self.sim.subaps_pos, self.sim.subap_masks, target_mode=self.sim_mode
        )
        z_wls, _ = reconstructor.reconstruct_wls(centroids, confidences, illum)
        
        # Dynamic GLAO component blending
        from vajra.controller import GLAODecomposer
        glao = GLAODecomposer()
        gl, ha = glao.decompose_slopes(centroids, illum)
        # Apply glao_weight to Ground Layer, rest to High Altitude
        controlled_slopes = glao_weight * gl + (1.0 - glao_weight) * ha
        z_glao, _ = reconstructor.reconstruct_wls(controlled_slopes, confidences, illum)
        
        # Target wavefront to correct
        z_target = 0.7 * z_wls + 0.3 * z_glao
        
        # Integrator update with RL gain
        from vajra.controller import ActuatorController
        controller = ActuatorController(reconstructor.G_full, self.sim.pupil_grid, reconstructor.zernike_basis)
        u_target = controller.zernike_to_command @ z_target
        
        # Dynamic integrator update
        self.current_commands = self.current_commands - loop_gain * u_target
        self.current_commands = np.clip(self.current_commands, -config.ACTUATOR_STROKE_LIMIT, config.ACTUATOR_STROKE_LIMIT)
        
        # Assemble observation
        obs_frame = np.concatenate([centroids.flatten(), confidences])
        self.history.append(obs_frame)
        if len(self.history) > self.history_len:
            self.history.pop(0)
        observation = np.concatenate(self.history)
        
        # Calculate Reward (Negative RMS wavefront error + action smoothness)
        dm_phase = self.sim.dm.phase_for(config.LAMBDA_SENSING)
        corrected_phase = (true_phase + dm_phase) * self.sim.pupil_mask
        rms_val = np.std(corrected_phase[self.sim.pupil_mask > 0])
        
        reward = -float(rms_val)
        
        # Penalize excessive action swings
        terminated = False
        truncated = (self.loop_time >= 0.1) # 100 ms loop episodes (100 steps)
        
        info = {
            "strehl": float(np.exp(-rms_val**2)),
            "rms": rms_val
        }
        return observation, reward, terminated, truncated, info


class PolicyNetwork(nn.Module):
    """SAC Actor Network generating mean and log_std for continuous parameter tuning."""
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(128, action_dim)
        self.log_std_head = nn.Linear(128, action_dim)
        
    def forward(self, state):
        x = self.shared(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample_action(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal = torch.randn_like(mean)
        action_t = mean + normal * std
        # Tanh squashing for bounded action spaces
        action = torch.tanh(action_t)
        
        # Log probability calculation
        log_prob = -0.5 * (((action_t - mean) / (std + 1e-8))**2 + 2 * log_std + np.log(2 * np.pi))
        log_prob -= torch.log(1.0 - action**2 + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class QNetwork(nn.Module):
    """SAC Critic Network measuring state-action Q value."""
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class SACAgent:
    """Soft Actor-Critic agent managing training updates and actions."""
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        
        # Policy Network
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=3e-4)
        
        # Critic Networks (Double Q-learning)
        self.q1 = QNetwork(state_dim, action_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim).to(device)
        self.q1_target = QNetwork(state_dim, action_dim).to(device)
        self.q2_target = QNetwork(state_dim, action_dim).to(device)
        
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=3e-4)
        
        self.entropy_alpha = 0.2
        self.gamma = 0.99
        self.tau = 0.005

    def select_action(self, state, evaluate=False):
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            if evaluate:
                mean, _ = self.policy(state_t)
                action = torch.tanh(mean).squeeze(0).cpu().numpy()
            else:
                action_t, _ = self.policy.sample_action(state_t)
                action = action_t.squeeze(0).cpu().numpy()
                
        # Denormalize action to matching bounds:
        # Loop gain: [0.1, 1.2]
        # GLAO: [0.0, 1.0]
        # Notch r: [0.90, 0.99]
        loop_gain = 0.1 + 0.55 * (action[0] + 1.0)
        glao_weight = 0.5 + 0.5 * action[1]
        notch_r = 0.90 + 0.045 * (action[2] + 1.0)
        
        # Clip to ensure validity
        loop_gain = np.clip(loop_gain, 0.1, 1.2)
        glao_weight = np.clip(glao_weight, 0.0, 1.0)
        notch_r = np.clip(notch_r, 0.90, 0.99)
        
        return np.array([loop_gain, glao_weight, notch_r], dtype=np.float32)

    def train_step(self, replay_buffer, batch_size=32):
        if len(replay_buffer) < batch_size:
            return
            
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        states_t = torch.tensor(states, dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.float32, device=device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(-1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(-1)
        
        # 1. Update Critics
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample_action(next_states_t)
            q1_next = self.q1_target(next_states_t, next_actions)
            q2_next = self.q2_target(next_states_t, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.entropy_alpha * next_log_probs
            q_target = rewards_t + (1 - dones_t) * self.gamma * q_next
            
        q1_val = self.q1(states_t, actions_t)
        q2_val = self.q2(states_t, actions_t)
        q1_loss = nn.MSELoss()(q1_val, q_target)
        q2_loss = nn.MSELoss()(q2_val, q_target)
        
        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()
        
        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()
        
        # 2. Update Policy
        new_actions, log_probs = self.policy.sample_action(states_t)
        q1_new = self.q1(states_t, new_actions)
        q2_new = self.q2(states_t, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.entropy_alpha * log_probs - q_new).mean()
        
        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()
        
        # 3. Soft updates of targets
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class SimpleReplayBuffer:
    """Stores experience tuples (s, a, r, s', done)."""
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.buffer = []
        self.ptr = 0

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.ptr] = experience
        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in idx:
            s, a, r, ns, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)
