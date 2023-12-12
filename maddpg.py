import torch as T
import torch.nn.functional as F
from torch.distributions import Categorical
from agent import Agent

from torchviz import make_dot

T.autograd.set_detect_anomaly(True)


class MADDPG:
    def __init__(self, actor_dims, n_agents, n_actions, 
                alpha=0.01, beta=0.01, fc1=64, 
                fc2=64, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.agents = {f"agent_{idx}":
            Agent(actor_dims[idx],  
                        n_actions, n_agents, idx, alpha=alpha, beta=beta,
                        chkpt_dir=chkpt_dir) for idx in range(self.n_agents)
        }

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents.values():
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents.values():
            agent.load_models()

    def choose_action(self, raw_obs):
        actions = {}
        for agent_idx, agent in self.agents.items():
            action = agent.choose_action(raw_obs[agent_idx])
            actions[agent_idx] = action
        return actions

    def learn(self, memory):
        if not memory.ready():
            print("memory is not ready")
            return

        actor_states, states, actions, rewards, \
        actor_new_states, states_, dones = memory.sample_buffer()

        device = list(self.agents.values())[0].actor.device

        # Pick up batch size and grid size
        batch_size, grid_size = states.shape[0], states.shape[-1]

        states = T.tensor(states, dtype=T.float).to(device).view(batch_size, -1, grid_size, grid_size) # join "n_agents" and "channel" dimension
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        #print(rewards)
        states_ = T.tensor(states_, dtype=T.float).to(device).view(batch_size, -1, grid_size, grid_size) # again, join the 2 dims
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        # For pulling out index and agent object, now we do not need a key "agent_{i}"
        for agent_idx, agent in enumerate(self.agents.values()):
            new_states = T.tensor(actor_new_states[agent_idx], dtype=T.float).clone().to(device ) # ?
            #new_pi = Categorical(agent.target_actor.forward(new_states)).sample()
            new_pi = agent.target_actor.forward(new_states)
            all_agents_new_actions.append(new_pi)

            mu_states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(F.one_hot(actions[:, agent_idx].long(), num_classes=self.n_actions))

        new_actions = T.cat([acts.unsqueeze(-1) for acts in all_agents_new_actions], dim=-1)
        mu = T.cat([acts.unsqueeze(-1) for acts in all_agents_new_mu_actions], dim=-1)
        old_actions = T.cat([acts.unsqueeze(-1) for acts in old_agents_actions],dim=-1)
        #print(new_actions.shape)
        #print(old_actions.shape)

        T.autograd.set_detect_anomaly(True)
        
        # For pulling out index and agent object, now we do not need a key "agent_{i}"
        for agent_idx, agent in enumerate(self.agents.values()):
            print("*********************************************************************")
            print(agent_idx)
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
            critic_value_[dones[:,0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()

            target = rewards[:,agent_idx] + agent.gamma*critic_value_
            critic_loss = F.mse_loss(target, critic_value)
            
            agent.critic.optimizer.zero_grad()
            print(critic_loss)
            print(target.shape)
            print(critic_value.shape)
            print(critic_value_.shape)
            critic_loss.backward(retain_graph=False)
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=False)
            agent.actor.optimizer.step()

            agent.update_network_parameters()
        
# EoF