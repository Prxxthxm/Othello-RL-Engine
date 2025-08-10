from othello import Othello
from game import Environment
from CNN_model import CNN_Player
import numpy as np
import random
import torch
import torch.nn as nn
from collections import deque

class Agent:
    def __init__(self,gamma=0.999,epsilon=1.0,lr=0.01,max_memory=10000,batch_size = 2):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.memory = deque(maxlen=max_memory)
        self.model = CNN_Player(out_channels=32, conv_size=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        self.loss_func = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def remember(self,state,action,reward,next_state,over):
        self.memory.append((state,action,reward,next_state,over))

    def action(self,state,legal_moves_mask):
        if random.random()<self.epsilon and len(legal_moves_mask)>0:
            move_index = (random.choice(legal_moves_mask))
            return 8 * move_index[0] + move_index[1]

        tensor_t = torch.tensor(state,dtype=torch.float32,device=self.device).unsqueeze(0)
        q_values = self.model(tensor_t).squeeze(0).detach().numpy()

        for i in range(len(legal_moves_mask)):
            if legal_moves_mask[i] == 0:
                q_values[i] = -np.inf
        return int(np.argmax(q_values))
    
    def replay(self):
        if len(self.memory)<self.batch_size:
            return
        batch = random.sample(self.memory,self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states_tensor = torch.tensor(np.stack(states), dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_value = self.model(states_tensor)
        q_predicted = q_value.gather(1,actions_tensor.unsqueeze(1)).squeeze(1)

        q_next = torch.zeros(self.batch_size,device=self.device)

        non_terminal_mask = (dones == 0)
        non_terminal_next_states = [s for s in next_states if s is not None]

        if non_terminal_next_states:
            non_terminal_next_states = torch.stack(non_terminal_next_states).to(self.device)
            q_next[non_terminal_mask] = self.model(non_terminal_next_states).detach().max(1)[0]

        q_target = rewards_tensor + (self.gamma * q_next * (1 - dones_tensor))

        loss = self.loss_func(q_target, q_predicted)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        env = Environment()
        for ep in range(1,50):
            env.reset()
            # if ep%10 == 0:
            #     print(ep)
            if ep%50 == 0:
                self.epsilon = max(self.epsilon*0.99,0.05)
            game = env.game
            while not game.is_game_over():
                state = env.get_state()
                legal_moves = game.get_legal_moves()
                if len(legal_moves) == 0:
                    game.current_turn *= -1
                action = self.action(state,legal_moves)

                next_state, reward, done = env.step(action=action)
                if done:
                    next_state = None
                self.remember(state,action,reward,next_state,done)
            self.replay()
agent = Agent()
agent.train()



    
        
        