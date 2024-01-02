import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Batch, HeteroData, Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import TransformerConv
from torch_geometric.data import DataLoader as GDataLoader


class GNN_Layer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=1, pool=False):
        super(GNN_Layer, self).__init__()
        self.in_dim=in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.transformer_conv = TransformerConv(in_dim, out_dim, heads=self.n_heads)
        self.act = F.relu
        self.pool = pool

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.transformer_conv(x, edge_index)
        x = self.act(x)

        if self.pool :
          x = global_mean_pool(x, data.batch)

        return x

class GraphVectorEncoder(nn.Module):
  def __init__(self, d_model, d_hidden, d_out, n_heads):
    super(GraphVectorEncoder, self).__init__()
    self.layer1 = GNN_Layer(d_model, d_hidden, n_heads)
    self.layer2 = GNN_Layer(d_hidden*n_heads, d_hidden, n_heads)
    self.layer3 = GNN_Layer(d_hidden*n_heads, d_out, 1, pool=True)

  def forward(self, data) :
    x = self.layer1(data)
    data = Batch(x=x, edge_index=data.edge_index, batch=data.batch)
    x = self.layer2(data)
    data = Batch(x=x, edge_index=data.edge_index, batch=data.batch)

    x = self.layer3(data)

    return x
