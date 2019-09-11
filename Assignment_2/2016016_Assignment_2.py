#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tqdm
import sys

from scipy.optimize import linprog


# # Question 2

# In[2]:


GRID_SIZE = 5
A, A_d = (0,1), (4,1)
B, B_d = (0,3), (2,3)

gamma = 0.9 # Given
pi_action = [0.25, 0.25, 0.25, 0.25] # Policy function
actions = [(-1,0), (1,0), (0,1), (0, -1)] # Set of actions i.e. left, right, up, down


# In[3]:


num_states = GRID_SIZE * GRID_SIZE # Every cell is a state


# In[4]:


def inside(GRID_SIZE, new_location):
    H,W = GRID_SIZE, GRID_SIZE
    if(new_location[0]<0):
        return False
    if(new_location[0]>=H):
        return False
    if(new_location[1]<0):
        return False
    if(new_location[1]>=W):
        return False
    return True


# In[5]:


def create_equations(actions, pi_action):
    mat_A = np.zeros((num_states, num_states)) # matrix (A)
    mat_B = np.zeros((num_states,1)) # vector (b), to solve Ax=b
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            pos = i * GRID_SIZE + j
            if (i,j) == A: # Special case if agent is at (0,1), new state = (4,1), reward = 10
                new_location = A_d[0] * GRID_SIZE + A_d[1]
                mat_A[pos, new_location] -= gamma 
                mat_B[pos] += 10
            elif (i,j) == B: # Special case if agent is at (0,3), new state = (2,3), reward = 5
                new_location = B_d[0] * GRID_SIZE + B_d[1]
                mat_A[pos, new_location] -= gamma
                mat_B[pos] += 5
            else:
                for k in range(len(actions)):
                    # For every action, agent goes to a state
                    # i.e. if action = right at state (0,0) then agent goes to (0,1) and has reward 0 with prob = 1
                    new_location = i + actions[k][0], j + actions[k][1]
                    if inside(GRID_SIZE, new_location):
                        pos_2 = new_location[0] * GRID_SIZE + new_location[1]
                        mat_A[pos,pos_2] -= pi_action[k] * gamma
                        mat_B[pos] += 0
                    else:
                        mat_A[pos,pos] -= pi_action[k] * gamma
                        mat_B[pos] += -1 * pi_action[k]
            mat_A[pos,pos] += 1
    return mat_A, mat_B


# In[6]:


A, b = create_equations(actions, pi_action)


# In[7]:


value_function = np.linalg.solve(A, b)
value_function = np.reshape(value_function, (GRID_SIZE,GRID_SIZE))
np.set_printoptions(precision=1)
value_function


# # Question 4

# In[8]:


np.set_printoptions(precision=8)


# In[9]:


GRID_SIZE = 5
A, A_d = (0,1), (4,1)
B, B_d = (0,3), (2,3)

gamma = 0.9 # Given
pi_action = [1, 1, 1, 1] # Policy function
actions = [(-1,0), (1,0), (0,1), (0, -1)] # Set of actions i.e. left, right, up, down


# In[10]:


num_states = GRID_SIZE * GRID_SIZE # Every cell is a state


# In[11]:


def create_equations(actions, pi_action):
    mat_A = np.zeros((num_states * len(actions), num_states)) # matrix (A)
    mat_B = np.zeros((num_states * len(actions),1)) # vector (b), to solve Ax=b
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            pos = i * GRID_SIZE + j
            if (i,j) == A: # Special case if agent is at (0,1), new state = (4,1), reward = 10
                new_location = A_d[0] * GRID_SIZE + A_d[1]
                for k in range(len(actions)):
                    mat_A[pos + k, new_location] -= gamma 
                    mat_B[pos + k] += 10
            elif (i,j) == B: # Special case if agent is at (0,3), new state = (2,3), reward = 5
                new_location = B_d[0] * GRID_SIZE + B_d[1]
                for k in range(len(actions)):
                    mat_A[pos + k, new_location] -= gamma
                    mat_B[pos + k] += 5
            else:
                for k in range(len(actions)):
                    new_location = i + actions[k][0], j + actions[k][1]
                    if inside(GRID_SIZE, new_location):
                        pos_2 = new_location[0] * GRID_SIZE + new_location[1]
                        mat_A[pos + k, pos_2] -= pi_action[k] * gamma
                        mat_B[pos + k] += 0
                    else:
                        mat_A[pos + k, pos] -= pi_action[k] * gamma
                        mat_B[pos + k] += -1 * pi_action[k]
            for k in range(len(actions)):
                mat_A[pos + k, pos] += 1
    return mat_A * -1, mat_B * -1


# In[12]:


mA, mb = create_equations(actions, pi_action)


# In[13]:


C = np.ones((num_states,1)) * -1
value_function = linprog(C, A_ub=mA, b_ub=mb)
value_function


# # Question 6

# In[14]:


GRID_SIZE = 4
terminal_states = [(0,0), (3,3)]
num_states = GRID_SIZE * GRID_SIZE

gamma = 1.0 # Given
actions = [(-1,0), (1,0), (0,1), (0, -1)] # Set of actions i.e. left, right, up, down


# In[15]:


def argmax(expected_return):
    indices = []
    max_val = np.max(expected_return)
    for i in range(expected_return.shape[0]):
        if expected_return[i] == max_val:
            indices.append(i)
    indices = np.array(indices)
    indices = np.reshape(indices, (indices.shape[0],1))
    indices = indices.astype(np.uint8)
    return indices


# ### Using policy interation

# In[16]:


V_s = np.zeros((num_states,1)) # Initialization of the value function
pi_action = np.ones((num_states,4)) * 0.25
tolerance = 0.1

policy_stable = False
while not policy_stable:
    
    # Policy evaluation
    while True:
        delta = 0.0
        v_s = np.copy(V_s)
        V_s = np.zeros(V_s.shape)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if (i,j) in terminal_states:
                    continue
                else:
                    s = i * GRID_SIZE + j
                    for k in range(len(actions)):
                        new_location = i + actions[k][0], j + actions[k][1]
                        if inside(GRID_SIZE, new_location):
                            new_s = new_location[0] * GRID_SIZE + new_location[1]
                            V_s[s] += (-1 + gamma * v_s[new_s]) * pi_action[s][k]
                        else:
                            V_s[s] += (-1 + gamma * v_s[s]) * pi_action[s][k]
                    delta = max(delta, np.absolute(V_s[s] - v_s[s]))
        if delta < tolerance:
            break
    
    print('Update value function for policy\n', V_s.reshape((GRID_SIZE,GRID_SIZE)))
            
    # Policy improvement
    old_policy = np.copy(pi_action)
    pi_action = np.zeros(pi_action.shape)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if (i,j) in terminal_states:
                continue
            else:
                s = i * GRID_SIZE + j
                expected_return = np.zeros((len(actions),1))
                for k in range(len(actions)):
                    new_location = i + actions[k][0], j + actions[k][1]
                    if inside(GRID_SIZE, new_location):
                        new_s = new_location[0] * GRID_SIZE + new_location[1]
                        expected_return[k] = (-1 + gamma * V_s[new_s])
                    else:
                        expected_return[k] = (-1 + gamma * V_s[s])
                best_actions = argmax(expected_return)
                pi_action[s][best_actions] = 1
                pi_action[s,:] /= np.sum(pi_action[s,:])
    print('Updated policy using new value function\n', pi_action)
    if np.all(pi_action == old_policy):
        print('\n\n***Found optimal policy***\n\n')
        break
    print('\n\n ***Iteration Complete, still searching for optimal policy***\n\n')

print('The optimal policy is\n')
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        print('{}: {}'.format((i,j), pi_action[i*GRID_SIZE+j]))
        
print("\nOrder of actions is down, up, right, left")


# ### Using value iteration

# In[17]:


V_s = np.zeros((num_states,1)) # Initialization of the value function
pi_action = np.ones((num_states,4)) * 0.25
tolerance = 0.1

while True:
    delta = 0.0
    v_s = np.copy(V_s)
    V_s = np.zeros(V_s.shape)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            s = i * GRID_SIZE + j
            if (i,j) in terminal_states:
                continue
            else:
                expected_return = np.zeros((len(actions),1))
                for k in range(len(actions)):
                    new_location = i + actions[k][0], j + actions[k][1]
                    if inside(GRID_SIZE, new_location):
                        s_new = new_location[0] * GRID_SIZE + new_location[1]
                        expected_return[k] = (-1 + gamma * v_s[s_new])
                    else:
                        expected_return[k] = (-1 + gamma * v_s[s])
                V_s[s] = np.max(expected_return)
                
                delta = max(delta, np.absolute(V_s[s] - v_s[s]))
    
    print('Update value function for policy\n', V_s.reshape((GRID_SIZE,GRID_SIZE)))
    
    if delta < tolerance:
        print('\n\n***Found oprimal value function***\n\n')
        break
    print('\n\n ***Iteration Complete, still searching for optimal value function***\n\n')
    
pi_action = np.zeros(pi_action.shape)
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        if (i,j) in terminal_states:
            continue
        else:
            s = i * GRID_SIZE + j
            expected_return = np.zeros((len(actions),1))
            for k in range(len(actions)):
                new_location = i + actions[k][0], j + actions[k][1]
                if inside(GRID_SIZE, new_location):
                    new_s = new_location[0] * GRID_SIZE + new_location[1]
                    expected_return[k] = (-1 + gamma * V_s[new_s])
                else:
                    expected_return[k] = (-1 + gamma * V_s[s])
            best_actions = argmax(expected_return)
            pi_action[s][best_actions] = 1
            pi_action[s,:] /= np.sum(pi_action[s,:])
            
print('The optimal policy is\n')
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        print('{}: {}'.format((i,j), pi_action[i*GRID_SIZE+j]))
        
print("\nOrder of actions is down, up, right, left")


# # Question 7

# ### Original Jack's Car Rental Problem with Max. Number of Cars = 10

# In[18]:


num_cars = 10
num_states = (num_cars + 1) * (num_cars + 1)

num_rentals_1 = 3
num_returns_1 = 3

num_rentals_2 = 4
num_returns_2 = 2

r_rent = 10
r_move = -2

max_move = 5

gamma = 0.9


# In[19]:


def poisson(x, lamb):
    p = np.exp(-lamb) * np.power(lamb, x) / np.math.factorial(x)
    return p


# In[20]:


def expected_return(state, action, value_function):
    e_return = 0.0
    e_return += r_move * np.absolute(action)
    new_state = min(state[0] - action, num_cars), min(state[1] + action, num_cars)
    new_state = int(new_state[0]), int(new_state[1])
    for i in range(0, new_state[0] + 1):
        for j in range(0, new_state[1] + 1):
            p_rent = poisson(i, num_rentals_1) * poisson(j, num_rentals_2)
            rent_reward = (i + j) * r_rent
            for k in range(0, 11):
                for l in range(0, 11):
                    p_return = poisson(k, num_returns_1) * poisson(l, num_returns_2)
                    final_state = new_state[0] - i + k, new_state[1] - j + l
                    final_state = min(final_state[0], num_cars), min(final_state[1], num_cars)
                    n_s = final_state[0] * (num_cars + 1) + final_state[1]
                    e_return += p_return * p_rent * (rent_reward + gamma * value_function[int(n_s)])
    
    return e_return


# In[ ]:


V_s = np.zeros((num_states,1)) # Value functions for all states
pi_action = np.zeros((num_cars + 1, num_cars + 1)) # Initial policy to not move any cars
tolerance = 1.0e-1

policy_stable = False
while not policy_stable:
    # Policy evaluation
    while True:
        delta = 0.0
        v_s = np.copy(V_s)
        V_s = np.zeros(V_s.shape)
        for i in range(num_cars + 1):
            for j in range(num_cars + 1):
                s = i * (num_cars + 1) + j
                V_s[s] = expected_return((i,j), pi_action[i,j], V_s)
                delta = max(delta, V_s[s] - v_s[s])
        if delta < tolerance:
            break
            
    print('Update value function for policy\n', V_s.reshape((num_cars + 1,num_cars + 1)))
    
    # Policy improvement
    old_policy = np.copy(pi_action)
    pi_action = np.zeros(pi_action.shape)
    for i in range(num_cars + 1):
        for j in range(num_cars + 1):
            s = i * (num_cars + 1) + j
            actions = np.arange(-max_move, max_move + 1)
            actions = np.reshape(actions, (actions.shape[0],1))
            all_return = np.zeros(actions.shape)
            for k in range(actions.shape[0]):
                if ((actions[k] >= 0 and i >= actions[k]) or (actions[k] < 0 and j >= np.absolute(actions[k]))):
                    all_return[k] = expected_return((i,j), actions[k], V_s)
                else:
                    all_return[k] = -np.inf
            pi_action[i,j] = actions[np.argmax(all_return)]
    
    print('Updated policy using new value function\n', pi_action.reshape(num_cars + 1, num_cars + 1))
    if np.all(pi_action == old_policy):
        print('\n\n***Found optimal policy***\n\n')
        break
    print('\n\n ***Iteration Complete, still searching for optimal policy***\n\n')


# In[ ]:


sns.set()
fig = plt.figure()
ax = sns.heatmap(pi_action.reshape(num_cars+1, num_cars+1))
fig.savefig('original jack problem.png')


# ### Modified problem

# In[ ]:


num_cars = 10
num_states = (num_cars + 1) * (num_cars + 1)

num_rentals_1 = 3
num_returns_1 = 3

num_rentals_2 = 4
num_returns_2 = 2

parking_space_limit = 5

r_rent = 10
r_move = -2
r_cars_more_than_5 = -4

max_move = 5

gamma = 0.9


# In[ ]:


def expected_return_modified(state, action, value_function):
    e_return = 0.0
    if action <=0:
        e_return += r_move * np.absolute(action)
    new_state = min(state[0] - action, num_cars), min(state[1] + action, num_cars)
    new_state = int(new_state[0]), int(new_state[1])
    if new_state[0] > parking_space_limit:
        e_return += r_cars_more_than_5
    if new_state[1] > parking_space_limit:
        e_return += r_cars_more_than_5
    for i in range(0, new_state[0] + 1):
        for j in range(0, new_state[1] + 1):
            p_rent = poisson(i, num_rentals_1) * poisson(j, num_rentals_2)
            rent_reward = (i + j) * r_rent
            for k in range(0, 11):
                for l in range(0, 11):
                    p_return = poisson(k, num_returns_1) * poisson(l, num_returns_2)
                    final_state = new_state[0] - i + k, new_state[1] - j + l
                    final_state = min(final_state[0], num_cars), min(final_state[1], num_cars)
                    n_s = final_state[0] * (num_cars + 1) + final_state[1]
                    e_return += p_return * p_rent * (rent_reward + gamma * value_function[int(n_s)])
    
    return e_return


# In[ ]:


V_s = np.zeros((num_states,1)) # Value functions for all states
pi_action = np.zeros((num_cars + 1, num_cars + 1)) # Initial policy to not move any cars
tolerance = 1.0e-1

policy_stable = False
while not policy_stable:
    # Policy evaluation
    while True:
        delta = 0.0
        v_s = np.copy(V_s)
        V_s = np.zeros(V_s.shape)
        for i in range(num_cars + 1):
            for j in range(num_cars + 1):
                s = i * (num_cars + 1) + j
                V_s[s] = expected_return((i,j), pi_action[i,j], V_s)
                delta = max(delta, V_s[s] - v_s[s])
        if delta < tolerance:
            break
            
    print('Update value function for policy\n', V_s.reshape((num_cars + 1,num_cars + 1)))
    
    # Policy improvement
    old_policy = np.copy(pi_action)
    pi_action = np.zeros(pi_action.shape)
    for i in range(num_cars + 1):
        for j in range(num_cars + 1):
            s = i * (num_cars + 1) + j
            actions = np.arange(-max_move, max_move + 1)
            actions = np.reshape(actions, (actions.shape[0],1))
            all_return = np.zeros(actions.shape)
            for k in range(actions.shape[0]):
                if ((actions[k] >= 0 and i >= actions[k]) or (actions[k] < 0 and j >= np.absolute(actions[k]))):
                    all_return[k] = expected_return((i,j), actions[k], V_s)
                else:
                    all_return[k] = -np.inf
            pi_action[i,j] = actions[np.argmax(all_return)]
    
    print('Updated policy using new value function\n', pi_action.reshape(num_cars + 1, num_cars + 1))
    if np.all(pi_action == old_policy):
        print('\n\n***Found optimal policy***\n\n')
        break
    print('\n\n ***Iteration Complete, still searching for optimal policy***\n\n')


# In[ ]:


sns.set()
ax = sns.heatmap(pi_action.reshape(num_cars+1, num_cars+1))


# In[ ]:


fig = plt.figure()
ax = sns.heatmap(pi_action.reshape(num_cars+1, num_cars+1))
fig.savefig('4.7.png')


# In[ ]:




