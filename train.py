import numpy as np
import random 
import torch
import torch.nn as nn 
import torch.optim as optim

from copy import deepcopy
from torch.autograd import Variable 

from dqn import ReplayMemory, Transition, hidden_unit, Q_learning
from gridworld import GridWorld
from util import Logger 


def train_DQN(epochs = 100, 
            gamma = 0.9, 
            epsilon = 1, 
            model = Q_learning, 
            model_options = ((64, [150, 150], 4, hidden_unit), {}), 
            optimizer = optim.RMSprop, 
            optimizer_options = ((), {'lr': 1e-2}), 
            loss = nn.MSELoss, 
            loss_options = ((), {}), 
            batch_size = 40, 
            memory = ReplayMemory, 
            memory_options = ((400,), {}), 
            logger = 'log.txt', 
            environ = GridWorld, 
            environ_options = ((), {}), 
            sync_every = 10, 
            ):
    
    logger = Logger(logger)
    
    q = model(*model_options[0], **model_options[1])
    q_hat = model(*model_options[0], **model_options[1])
    
    optimizer = optimizer(q.parameters(), *optimizer_options[0], **optimizer_options[1])
    loss = loss(*loss_options[0], **loss_options[1])
    memory = memory(*memory_options[0], **memory_options[1])
    
    for i in range(epochs):
        logger.write(i)

        step = 0 

        # sampling phase 
        while not memory.is_full():
            ev = environ(*environ_options[0], **environ_options[1])
            actions = ev.actions() 

            while not ev.finished():
                state = ev.state_tensor()
                q_value = q(state)

                if random.random() < epsilon:
                    action = random.choice(actions)
                else:
                    action = np.argmax(q_value.data)
                    assert action in actions 
                
                ev.make_move(action)
                
                reward = ev.get_reward()
                new_state = ev.state_tensor()
                
                memory.push(state.data, action, new_state.data, reward)

        # training_phase
        for j in range(len(memory) // batch_size):
            batch = memory.sample(batch_size)
            
            state_batch = Variable(torch.cat(batch.state))
            action_batch = Variable(torch.LongTensor(batch.action)).view(-1,1)
            new_state_batch = Variable(torch.cat(batch.new_state))
            reward_batch = Variable(torch.FloatTensor(batch.reward))        

            q_val = q(state_batch)
            state_action_values = q_val.gather(1, action_batch)
            
            q_hat_val = q_hat(new_state_batch)
            max_q_hat = q_hat_val.max(1)[0]
            
            
            non_final_mask = (reward_batch == -1)

            y = reward_batch
            y[non_final_mask] += gamma * max_q_hat[non_final_mask]
            y = y.view(-1, 1)

            loss_val = loss(state_action_values, y)
            

            optimizer.zero_grad()
            loss_val.backward()
            for p in q_hat.parameters():
                p.grad.data.clamp_(-1, 1)
            optimizer.step()
            step += 1 

            if step % sync_every == 0:
                q_hat.load_state_dict(q.state_dict())
            
        if epsilon > 0.1:
            epsilon -= (1/epochs)
    
    return q 

def test_model(model, 
            environ = GridWorld, 
            environ_options = ((), {}), ):
    ev = environ(*environ_options[0], **environ_options[1])

    step = 0
    ev.display()

    while not ev.finished():
        q_val = model(ev.state_tensor())
        action = np.argmax(q_val.data)
        print(action)
        state = ev.make_move(action)
        
        ev.display()
        reward = ev.get_reward()
        step += 1 

        if step > 10:
            print('failed')
            break 

    print(reward)



if __name__ == '__main__':
    q = train_DQN(epochs = 1000)
    test_model(q)