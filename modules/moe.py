import torch
import torch.nn.functional as F
from torch import nn


class SparseMoE(nn.Module):
    def __init__(self,input_gate_size, input_embed_size, num_experts, output_expert_size):
        super().__init__()
        self.gate_network = nn.Sequential(
            nn.LayerNorm(input_gate_size, elementwise_affine=False),
            nn.Linear(input_gate_size, input_gate_size*2),
            nn.Linear(input_gate_size*2, num_experts),
            nn.LogSoftmax(dim=1)
        )
        self.expert_networks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_embed_size, output_expert_size*2),
                    # nn.BatchNorm1d(output_expert_size*2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.Linear(output_expert_size*2, output_expert_size)
                )
                for _ in range(num_experts)
            ])
        self.combine_experts = nn.Linear(output_expert_size, 1)

    def forward(self,x, y):
        gate_outputs = self.gate_network(x)
        expert_outputs = [net(y) for net in self.expert_networks]
        combined_output = torch.stack(expert_outputs, dim=1) * gate_outputs.unsqueeze(-1)
        combined_output = combined_output.sum(dim=1)
        final_output = self.combine_experts(combined_output) 

        return final_output
