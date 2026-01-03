import torch
import torch.nn as nn

class DummyMemoryStub(nn.Module):
    def __init__(self, hidden_size, query_dim=128, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.query_proj = nn.Linear(hidden_size, query_dim, **factory_kwargs)
    
    def forward(self, hidden_states):
        # We project the last token's hidden state to a query vector
        return self.query_proj(hidden_states[:, -1, :])
