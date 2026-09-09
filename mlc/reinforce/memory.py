from collections import deque
import numpy as np
import torch
import random

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = [0.0] * (2 * capacity - 1)
        self.data = [None] * capacity
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

        if self.n_entries < self.capacity:
            self.n_entries += 1
        self.write = (self.write + 1) % self.capacity

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - (self.capacity - 1)
        return idx, self.tree[idx], self.data[data_idx]

    def total(self):
        return self.tree[0]

class PriorityQueue:
    def __init__(self, capacity, n_step = 5, gamma = 0.99):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=n_step)
        self.max_priority = 1.0  # Initialize: Prevent all priority = 0
        self.gamma = gamma

    def store(self,state, action, reward, next_state, done):
            # Add n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # If it reaches n step, process real transition
        if len(self.n_step_buffer) == self.n_step:
            s, a, _, _, _ = self.n_step_buffer[0]
            R, s_n, done_n, n = self._compute_n_step_return()
            transition = (s, a, R, s_n, done_n, n)
            self.tree.add(self.max_priority, transition)
            self.n_step_buffer.popleft()

        # If episode is done, clear
        if done:
            while len(self.n_step_buffer) > 0:
                s, a, _, _, _ = self.n_step_buffer[0]
                R, s_n, done_n, n = self._compute_n_step_return()
                transition = (s, a, R, s_n, done_n, n)
                self.tree.add(self.max_priority, transition)
                self.n_step_buffer.popleft()
        
    def _compute_n_step_return(self):
        R = 0
        for idx, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            R += (self.gamma ** idx) * r
            if d:
                break
        s_n = self.n_step_buffer[idx][3]
        done_n = self.n_step_buffer[idx][4]
        return R, s_n, done_n, idx

    def sample(self,batch_size,beta = 0.4):
        indices = []
        priorities = []
        transitions = []

        # Total priority
        total = self.tree.total()
        if total == 0:
          raise ValueError("Cannot sample from an empty SumTree with total priority 0.")

        # Segment
        segment = total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)  # Uniform sampling in [a, b]
            idx, p, data = self.tree.get(s)
            priorities.append(p)
            transitions.append(data)
            indices.append(idx)

        # normalize IS weights
        probs = np.array(priorities) / total
        weights = (self.tree.n_entries * probs) ** (-beta)
        weights /= weights.max()  # normalize

        # Unpack batch
        states, actions, rewards, next_states, dones, ns = zip(*transitions)

        # Check your output
        return (
            states, # 1
            actions, # 2
            rewards, # 3
            next_states, # 4
            dones, # 5
            ns, # 6
            indices, # 7
            weights # 8
        )

    def update_priorities(self,indices, new_priorities):
        for idx, priority in zip(indices, new_priorities):
            epsilon = 1e-6
            priority = np.clip(float(priority), epsilon, None) # 👈 Prevent priority becomes 0
            self.tree.update(int(idx), priority) # 👈 Sent back to SumTree
            self.max_priority = max(self.max_priority, priority) # 👈 Remain max for new transition

    def __len__(self):
      return self.tree.n_entries

    def clear(self):
      self.tree = SumTree(self.tree.capacity)  # Build a new SumTree
      self.n_step_buffer.clear()               # Clear multi-step buffer


class MultistepReplayBuffer:
    def __init__(self, capacity = 30000, n_step = 10, gamma=0.99):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=capacity)
        self.n_step_buffer = deque(maxlen=n_step)
        self.position = 0

    def store(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done)) # retira o estado mais antigo
        if len(self.n_step_buffer) == self.n_step:
            s, a, _, _, _ = self.n_step_buffer[0]
            R, s_n, d_n, n = self._compute_n_return()
            self.buffer.append((s, a, R, s_n, d_n, n))
            
        if done:
            # Se o buffer estava cheio, o elemento 0 já foi salvo no `if` acima, 
            # então descartamos ele antes de salvar o restante.
            if len(self.n_step_buffer) == self.n_step:
                self.n_step_buffer.popleft()
                
            # Salva todos os estados finais (os últimos n-1 passos)
            while len(self.n_step_buffer) > 0:
                s, a, _, _, _ = self.n_step_buffer[0]
                R, s_n, d_n, n = self._compute_n_return()
                self.buffer.append((s, a, R, s_n, d_n, n))
                self.n_step_buffer.popleft()

    def _compute_n_return(self):
        R = 0
        for i,(_, _, r, _, d) in enumerate(self.n_step_buffer):
            R += (self.gamma ** i) * r
            if d: break
        d_n = self.n_step_buffer[i][4] # é d
        s_n = self.n_step_buffer[i][3]
        return R, s_n, d_n, i

    def clear(self):
        self.buffer.clear()
        self.n_step_buffer.clear()
        

    def sample(self, batch_size):    
        batch = random.sample(self.buffer, batch_size) #lista com batch_size tuplas de 6 tensores
        states, actions, rewards, next_states, dones, ns = zip(*batch) #tupla de 6 tuplas de 64 tensores
        return (states, actions, rewards, next_states, dones, ns)

    def __len__(self):
        return len(self.buffer)