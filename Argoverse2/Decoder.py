import torch.nn as nn

class DenseLayer(nn.Module):
  def __init__(self, in_dim, out_dim, alpha=0.1, act='relu'):
    super(DenseLayer, self).__init__()
    self.alpha = alpha
    self.fc = nn.Linear(in_dim, out_dim)
    self.bn = nn.BatchNorm1d(out_dim)
    self.act = nn.ReLU()
    self.act_name = act
    self.do = nn.Dropout(self.alpha)

  def forward(self, x):
    x = self.fc(x)
    if len(x) > 1:
      x = self.bn(x)
    
    if self.act_name=='relu': 
      x = self.act(x)
      x = self.do(x)
    
    return x


class Decoder(nn.Module):
  def __init__(self, in_dim, hidden_dim, out_dim):
    super(Decoder, self).__init__()

    self.fc1 = DenseLayer(in_dim, hidden_dim, alpha=0.2)
    
    self.fc2 = DenseLayer(hidden_dim, hidden_dim, alpha=0.1)
    self.fc3 = DenseLayer(hidden_dim, out_dim, act=None)

  def forward(self, x):
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    return x
