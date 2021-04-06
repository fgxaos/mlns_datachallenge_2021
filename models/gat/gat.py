### LIBRARIES ###
# Global libraries
import numpy as np

import torch
import torch.nn as nn

# Custom libraries
from models.gat.gat_layer import GraphAttention

### UTILS FUNCTION ###
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.414)


### CLASS DEFINITION ###
class GAT(nn.Module):
    """Graph Attention Network."""

    def __init__(
        self, input_dim, hidden_dims, output_dim, n_heads, dropout=0.5, device="cpu"
    ):
        """Initiates a GAT model.

        Args:
            input_dim: int
                dimension of input node features
            hidden_dims: List[int]
                dimension of hidden layers. Must be non-empty
            output_dim: int
                dimension of output node features
            n_heads: List[int]
                number of attention heads in each hidden layer
                and output layer. Must be non-empty
            dropout: float
                dropout rate
            device: str
                specifies which device to use
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.device = device
        self.n_layers = len(hidden_dims) + 1

        dims = (
            [input_dim]
            + [d * nh for (d, nh) in zip(hidden_dims, n_heads[:-1])]
            + [output_dim * n_heads[-1]]
        )
        in_dims = dims[:-1]
        out_dims = [d // nh for (d, nh) in zip(dims[1:], n_heads)]

        self.attn = nn.ModuleList(
            [
                GraphAttention(i, o, nh, dropout, device)
                for (i, o, nh) in zip(in_dims, out_dims, n_heads)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(dim) for dim in dims[1:-1]])

        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()

    def forward(self, features, node_layers, mappings, rows):
        """Forward function of the Graph attention network.

        Args:
            features: torch.Tensor (n' x input_dim)
                input node features
            node_layers: List[np.array]
                `node_layers[i]` corresponds to the array of the
                nodes in the i-th layer of the computation graph
            mappings: List[Dict{}]
                `mappings[i]` maps node `v` (in [0, |V|-1])
                in `node_layers[i]` to its position
                in `node_layers[i]`.
            rows: np.array
                `rows[i]` corresponds to the neighbors of node `i`
        Returns:
            out: torch.Tensor (len(node_layers[-1]) x output_dim)
                tensor of output node features
        """
        out = features

        for k in range(self.n_layers):
            nodes = node_layers[k + 1]
            mapping = mappings[k]
            init_mapped_nodes = np.array(
                [mappings[0][v] for v in nodes], dtype=np.int64
            )
            cur_rows = rows[init_mapped_nodes]
            out = self.dropout(out)
            out = self.attn[k](out, nodes, mapping, cur_rows)

            if k + 1 < self.n_layers:
                out = [self.elu(o) for o in out]
                out = torch.cat(tuple(out), dim=1)
                out = self.bns[k](out)
            else:
                out = torch.cat(tuple([x.flatten().unsqueeze(0) for x in out]), dim=0)
                out = out.mean(dim=0).reshape(len(nodes), self.output_dim)

        return out
