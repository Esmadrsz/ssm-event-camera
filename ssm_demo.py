import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# SSM implementation - inspired by Gu et al. 2021 (S4 paper)
# A, B, C matrices remind me of my thesis work on dynamical systems

class SimpleSSM(nn.Module):
    def __init__(self, input_dim, state_dim, output_dim):
        super().__init__()
        # state transition - like the A matrix in x'(t) = Ax(t) + Bu(t)
        self.A = nn.Linear(state_dim, state_dim, bias=False)
        # input projection
        self.B = nn.Linear(input_dim, state_dim, bias=False)
        # output projection - maps hidden state to output
        self.C = nn.Linear(state_dim, output_dim, bias=False)
        self.state_dim = state_dim
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        # initialize hidden state as zeros
        h = torch.zeros(batch_size, self.state_dim)
        outputs = []
        
        for t in range(seq_len):
            # update hidden state at each time step
            # tanh keeps values bounded - tried relu first but tanh worked better
            h = torch.tanh(self.A(h) + self.B(x[:, t, :]))
            y = self.C(h)
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)

# simulate event stream data
# real datasets: N-MNIST, DVS-Gesture - using synthetic for now
def generate_event_stream(seq_len=100):
    t = np.linspace(0, 4*np.pi, seq_len)
    # sine wave + gaussian noise to mimic event camera output
    events = np.sin(t) + 0.1 * np.random.randn(seq_len)
    return events

# run
model = SimpleSSM(input_dim=1, state_dim=16, output_dim=1)
events = generate_event_stream()

# need to add batch and feature dimensions for pytorch
x = torch.tensor(events, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

with torch.no_grad():
    output = model(x)

# plot input vs output
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

ax1.plot(events, color='blue', label='Event Stream Input')
ax1.set_title('Simulated Event-Based Camera Data')
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Event Signal')
ax1.legend()
ax1.grid(True)

ax2.plot(output.squeeze().numpy(), color='red', label='SSM Output')
ax2.set_title('SSM Model Output')
ax2.set_xlabel('Time Step')
ax2.set_ylabel('Processed Signal')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('ssm_output.png')
plt.show()
print("Done! Graph saved as ssm_output.png")
