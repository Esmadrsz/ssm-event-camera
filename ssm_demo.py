import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class SimpleSSM(nn.Module):
    def __init__(self, input_dim, state_dim, output_dim):
        super().__init__()
        self.A = nn.Linear(state_dim, state_dim, bias=False)
        self.B = nn.Linear(input_dim, state_dim, bias=False)
        self.C = nn.Linear(state_dim, output_dim, bias=False)
        self.state_dim = state_dim
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.state_dim)
        outputs = []
        for t in range(seq_len):
            h = torch.tanh(self.A(h) + self.B(x[:, t, :]))
            y = self.C(h)
            outputs.append(y)
        return torch.stack(outputs, dim=1)

# Simulated event stream
def generate_event_stream(seq_len=100):
    t = np.linspace(0, 4*np.pi, seq_len)
    events = np.sin(t) + 0.1 * np.random.randn(seq_len)
    return events

# Run model
model = SimpleSSM(input_dim=1, state_dim=16, output_dim=1)
events = generate_event_stream()
x = torch.tensor(events, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
with torch.no_grad():
    output = model(x)

# Plot
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