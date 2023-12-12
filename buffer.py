import numpy as np


class MultiAgentReplayBuffer:
    def __init__(self, max_size, actor_dims,
                n_actions, n_agents, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.actor_dims = actor_dims # (channel, grid_size, grid_size)
        self.batch_size = batch_size
        self.n_actions = n_actions

        # Initialize memory for states, next states, actions, rewards and dones
        # States and next states are stores as 4D arrays
        self.actor_state_memory = [np.zeros((self.mem_size, *dims)) for dims in actor_dims]
        self.actor_new_state_memory = [np.zeros((self.mem_size, *dims)) for dims in actor_dims]
        
        # Memory for combined state for critic
        self.state_memory = np.zeros((self.mem_size, n_agents, *actor_dims[0]))
        self.new_state_memory = np.zeros((self.mem_size, n_agents, *actor_dims[0]))
        
        self.actor_action_memory = np.zeros((self.mem_size, self.n_agents))

        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)

    def store_transition(self, raw_obs, action, reward, raw_obs_, done):
        index = self.mem_cntr % self.mem_size

        # Store each agent's observations and actions
        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx] # agent's obs is (channel, height, width)
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
        
        #print(index)
        self.actor_action_memory[index] = action
        self.state_memory[index] = raw_obs
        self.new_state_memory[index] = raw_obs_
        
        # Store rewards and terminal flags
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False) # random indexes for pulling batch of data

        actor_states_batch = [self.actor_state_memory[i][batch] for i in range(self.n_agents)]
        actor_new_states_batch = [self.actor_new_state_memory[i][batch] for i in range(self.n_agents)]
        action_batch = self.actor_action_memory[batch]

        combined_states_batch = self.state_memory[batch]
        combined_new_states_batch = self.state_memory[batch]

        rewards_batch = self.reward_memory[batch]
        terminal_batch = self.terminal_memory[batch]

        return actor_states_batch, combined_states_batch, action_batch, rewards_batch, actor_new_states_batch, combined_new_states_batch, terminal_batch

    def ready(self):
        return self.mem_cntr >= self.batch_size

# EoF