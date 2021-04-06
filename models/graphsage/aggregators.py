### LIBRARIES ###
# Global libraries
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

### CLASS DEFINITIONS ###
class Aggregator(nn.Module):
    """Aggregator class."""

    def __init__(self, input_dim=None, output_dim=None, device="cpu"):
        """Initiates the Aggregator class.

        Args:
            input_dim: int | None
                dimension of input node features.
                Used for definition of fully-connected layers
                in pooling aggregators
            output_dim: int | None
                dimension of output node features.
                Used for definition of fully-connected layers
                in pooling aggregators
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

    def forward(self, features, nodes, mapping, rows, num_samples=25):
        """Forward function of the Aggregator module.

        Args:
            features: torch.Tensor (n' x input_dim)
                input node features
            nodes: np.array
                nodes in the current layer of the computation graph
            mapping: Dict
                maps node `v` to its positino in the layer of nodes
                in the computation graph before nodes
            rows: np.array
                neighbors of nodes present in `nodes`
            num_samples: int
                number of neighbors to sample while aggregating
        Returns:
            out: torch.Tensor (len(nodes) x output_dim)
                output node features
        """
        _choice, _len, _min = np.random.choice, len, min
        mapped_rows = [
            np.array([mapping[v] for v in row], dtype=np.int64) for row in rows
        ]
        if num_samples == -1:
            sampled_rows = mapped_rows
        else:
            sampled_rows = [
                _choice(row, _min(_len(row), num_samples), _len(row) < num_samples)
                for row in mapped_rows
            ]

        n = _len(nodes)
        if self.__class__.__name__ == "LSTMAggregator":
            out = torch.zeros(n, 2 * self.output_dim).to(self.device)
        else:
            out = torch.zeros(n, self.output_dim).to(self.device)
        for i in range(n):
            if _len(sampled_rows[i]) != 0:
                out[i, :] = self._aggregate(features[sampled_rows[i], :])

        return out

    def _aggregate(self, features):
        raise NotImplementedError


class MeanAggregator(Aggregator):
    def _aggregate(self, features):
        """Mean aggregation.

        Args:
            features: torch.Tensor
                input features
        Returns:
            aggregated features
        """
        return torch.mean(features, dim=0)


class PoolAggregator(Aggregator):
    def __init__(self, input_dim, output_dim, device="cpu"):
        """Initiates the PoolAggregator class.

        Args:
            input_dim: int
                dimension of input node features
            output_dim: int
                dimension of output node features
        """
        super().__init__(input_dim, output_dim, device)

        self.fc1 = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def _aggregate(self, features):
        """Pooling aggregation.

        Args:
            features: torch.Tensor
                input features
        returns:
            aggregated features
        """
        out = self.relu(self.fc1(features.float()))
        return self._pool_fn(out)

    def _pool_fn(self, features):
        raise NotImplementedError


class MaxPoolAggregator(PoolAggregator):
    def _pool_fn(self, features):
        """Max pooling aggregation.

        Args:
            features: torch.Tensor
                input features
        Returns:
            aggregated features
        """
        return torch.max(features, dim=0)[0]


class MeanPoolAggregator(PoolAggregator):
    def _pool_fn(self, features):
        """Mean pooling aggregation.

        Args:
            features: torch.Tensor
                input features
        Returns:
            aggregated feature
        """
        return torch.mean(features, dim=0)[0]


class LSTMAggregator(Aggregator):
    def __init__(self, input_dim, output_dim, device="cpu"):
        """Implements a LSTM aggregator.

        Args:
            input_dim: int
                dimension of input node features
            output_dim: int
                dimension of output node features
        """
        super().__init__(input_dim, output_dim, device)

        self.lstm = nn.LSTM(input_dim, output_dim, bidirectional=True, batch_first=True)

    def _aggregate(self, features):
        """LSTM aggregation.

        Args:
            features: torch.Tensor
                input features
        Returns:
            aggregated features
        """
        perm = np.random.permutation(np.arange(features.shape[0]))
        features = features[perm, :]
        features = features.unsqueeze(0)

        out, _ = self.lstm(features)
        out = out.squeeze(0)
        out = torch.sum(out, dim=0)

        return out
