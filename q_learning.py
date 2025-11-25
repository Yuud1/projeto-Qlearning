import numpy as np
import random
from typing import Tuple
from blackjack_env import BlackjackEnv, Action


class QLearningAgent:
    def __init__(
        self,
        num_states: int = 18,
        num_actions: int = 2,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.Q = np.zeros((num_states, num_actions))
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_episodes = 0
        
    def get_action(self, state: int, training: bool = True) -> Action:
        if training and random.random() < self.epsilon:
            return random.choice(list(Action))
        else:
            action_idx = np.argmax(self.Q[state, :])
            return Action(action_idx)
    
    def update(self, state: int, action: Action, reward: float, next_state: int, done: bool):
        action_idx = action.value
        current_q = self.Q[state, action_idx]
        
        if done:
            target_q = reward
        else:
            max_next_q = np.max(self.Q[next_state, :])
            target_q = reward + self.gamma * max_next_q
        
        self.Q[state, action_idx] = current_q + self.alpha * (target_q - current_q)
    
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train_episode(self, env: BlackjackEnv) -> Tuple[float, int]:
        state = env.reset()
        total_reward = 0.0
        steps = 0
        
        while not env.done:
            action = self.get_action(state, training=True)
            
            next_state, reward, done, info = env.step(action)
            
            self.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        self.total_episodes += 1
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)
        
        self.decay_epsilon()
        
        return total_reward, steps
    
    def train(self, env: BlackjackEnv, num_episodes: int = 10000):
        print(f"Iniciando treinamento por {num_episodes} episódios...")
        
        for episode in range(num_episodes):
            reward, steps = self.train_episode(env)
            
            if (episode + 1) % 1000 == 0:
                avg_reward = np.mean(self.episode_rewards[-1000:])
                avg_steps = np.mean(self.episode_lengths[-1000:])
                print(f"Episódio {episode + 1}/{num_episodes} | "
                      f"Recompensa média: {avg_reward:.3f} | "
                      f"Passos médios: {avg_steps:.2f} | "
                      f"Epsilon: {self.epsilon:.3f}")
    
    def get_policy(self) -> np.ndarray:
        return np.argmax(self.Q, axis=1)
    
    def get_stats(self) -> dict:
        if len(self.episode_rewards) == 0:
            return {}
        
        recent_rewards = self.episode_rewards[-1000:] if len(self.episode_rewards) >= 1000 else self.episode_rewards
        
        return {
            'total_episodes': self.total_episodes,
            'avg_reward_recent': np.mean(recent_rewards),
            'win_rate': np.mean([r > 0 for r in recent_rewards]),
            'epsilon': self.epsilon
        }

