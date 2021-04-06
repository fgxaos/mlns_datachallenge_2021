### LIBRARIES ###
# Global libraries
import sys

import numpy as np

import torch
import torch.nn as nn

# Custom libraries
from models.graphsage.aggregators import (
    MeanAggregator,
    LSTMAggregator,
    MaxPoolAggregator,
    MeanPoolAggregator,
)

### UTILS FUNCTIONS ###
def get_agg_class(agg_class):
    """Gets the Aggregator.

    Args:
        agg_class: str
            name of the aggregator class
    Returns:
        Aggregator class
    """
    return getattr(sys.modules[__name__], agg_class)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.414)


### CLASS DEFINITION ###


class GraphSage(nn.Module):
    """Implementation of the GraphSage model."""

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim,
        device,
        dropout=0.5,
        agg_class=MaxPoolAggregator,
        num_samples=25,
    ):
        """Initiates the GraphSage model.

        Args:
            input_dim: int
                dimension of the input node features
            hidden_dims: List[int]
                dimension of hidden layers
            output_dim: int
                dimension of the output features
            device: str
                device to use
            dropout: float
                dropout rate
            agg_class: Aggregator
                one of the defined aggregator classess
            num_samples: int
                number of neighbors to sample while aggregating
        """
        super(GraphSage, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.agg_class = agg_class
        self.num_samples = num_samples
        self.device = device
        self.num_layers = len(hidden_dims) + 1

        self.aggregators = nn.ModuleList([agg_class(input_dim, input_dim, device)])
        self.aggregators.extend([agg_class(dim, dim, device) for dim in hidden_dims])

        c = 3 if agg_class == LSTMAggregator else 2
        self.fcs = nn.ModuleList([nn.Linear(c * input_dim, hidden_dims[0])])
        self.fcs.extend(
            [
                nn.Linear(c * hidden_dims[i - 1], hidden_dims[i])
                for i in range(1, len(hidden_dims))
            ]
        )
        self.fcs.extend([nn.Linear(c * hidden_dims[-1], output_dim)])

        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for hidden_dim in hidden_dims]
        )

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, features, node_layers, mappings, rows):
        """Forward function of the GraphSage model.

        Args:
            features: torch.Tensor (n' x input_dim)
                input node features
            node_layers: List[np.array]
                nodes in the layers of the computation graph
            mappings: List[Dict]
                maps node `v` in `node_layers[i]` to its position in `node_layers[i]`
            rows: np.array
                neighbors of each node
        Returns:
            out: torch.Tensor (len(node_layers[-1]) x output_dim)
                output node features
        """
        out = features
        for k in range(self.num_layers):
            nodes = node_layers[k + 1]
            mapping = mappings[k]
            init_mapped_nodes = np.array(
                [mappings[0][v] for v in nodes], dtype=np.int64
            )
            cur_rows = rows[init_mapped_nodes]
            aggregate = self.aggregators[k](
                out, nodes, mapping, cur_rows, self.num_samples
            )
            cur_mapped_nodes = np.array([mapping[v] for v in nodes], dtype=np.int64)
            out = torch.cat((out[cur_mapped_nodes, :], aggregate), dim=1)
            out = self.fcs[k](out)
            if k + 1 < self.num_layers:
                out = self.relu(out)
                out = self.bns[k](out)
                out = self.dropout(out)
                out = out.div(out.norm(dim=1, keepdim=True) + 1e-6)

        return out