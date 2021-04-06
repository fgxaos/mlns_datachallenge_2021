### LIBRARIES ###
# Global libraries
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

### CLASS DEFINITION ###
class GraphAttention(nn.Module):
    """Graph attention module for the GAT model."""

    def __init__(self, input_dim, output_dim, n_heads, dropout=0.5, device="cpu"):
        """Initiates a Graph attention layer.

        Args:
            input_dim: int
                dimension of input node features
            output_dim: int
                dimension of output features after each attention head
            n_heads: int
                number of attention heads
            dropout: float
                dropout rate
            device: str
                device to use
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads

        self.fcs = nn.ModuleList(
            [nn.Linear(input_dim, output_dim) for _ in range(n_heads)]
        )
        self.a = nn.ModuleList([nn.Linear(2 * output_dim, 1) for _ in range(n_heads)])

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=0)
        self.leakyrelu = nn.LeakyReLU()

        self.device = device

    def forward(self, features, nodes, mapping, rows):
        """Forward function of the GraphAttention module.

        Args:
            features: torch.Tensor (n' x input_dim)
                input node features
            nodes: np.array
                nodes in the current layer of the computation graph
            mapping: Dict{}
                maps node `v` (from 0 to |V|-1) to its position in
                the layer of nodes in the computatino graph
                before nodes
            rows: np.array
                `rows[i]` represents the neighbors of node `i`
                which is present in `nodes`
        Returns:
            out: List[torch.Tensor] (len(nodes) x input_dim)
                output node features
        """
        features = features.float()
        nprime = features.shape[0]
        rows = [np.array([mapping[v] for v in row], dtype=np.int64) for row in rows]
        sum_degs = np.hstack(([0], np.cumsum([len(row) for row in rows])))
        mapped_nodes = [mapping[v] for v in nodes]
        indices = (
            torch.LongTensor(
                [[v, c] for (v, row) in zip(mapped_nodes, rows) for c in row]
            )
            .t()
            .to(self.device)
        )

        out = []
        for k in range(self.n_heads):
            h = self.fcs[k](features)

            nbr_h = torch.cat(tuple([h[row] for row in rows if len(row) > 0]), dim=0)
            self_h = torch.cat(
                tuple(
                    [
                        h[mapping[nodes[i]]].repeat(len(row), 1)
                        for (i, row) in enumerate(rows)
                        if len(row) > 0
                    ]
                ),
                dim=0,
            )
            attn_h = torch.cat((self_h, nbr_h), dim=1)

            e = self.leakyrelu(self.a[k](attn_h))

            alpha = [self.softmax(e[lo:hi]) for (lo, hi) in zip(sum_degs, sum_degs[1:])]
            alpha = torch.cat(tuple(alpha), dim=0)
            alpha = alpha.squeeze(1)
            alpha = self.dropout(alpha)

            adj = torch.sparse.FloatTensor(indices, alpha, torch.Size([nprime, nprime]))
            out.append(torch.sparse.mm(adj, h)[mapped_nodes])

        return out
