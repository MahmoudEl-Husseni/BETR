from config import *
from Decoder import *
from transformer import *
from graph_layers import *
from utils.geometry import *

import torch.nn.functional as F

from torch_geometric.data import Batch, HeteroData, Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import TransformerConv
from torch_geometric.data import DataLoader as GDataLoader


class VectorEncoder(nn.Module):
  def __init__(self, d_model, n_heads, hidden_dim, hidden_nheads, output_dim):
    super(VectorEncoder, self).__init__()
    self.d_model = d_model
    self.n_heads = n_heads
    self.hidden_dim = hidden_dim
    self.hidden_nheads = hidden_nheads
    self.output_dim = output_dim


    self.p_enc = PositionalEncoding(self.d_model, max_len=60)
    self.attn_layer1 = TransformerEncoderLayer(self.d_model, self.n_heads, self.hidden_dim, self.hidden_dim)
    self.attn_layer2 = TransformerEncoderLayer(self.hidden_dim, self.hidden_nheads, self.hidden_dim, self.hidden_dim)
    self.attn_layer3 = TransformerEncoderLayer(self.hidden_dim, self.hidden_nheads, self.hidden_dim, self.output_dim)

  def forward(self, x):
    # Add Position Encoding
    x = self.p_enc(x)

    # Pass to three transformer layers
    x, self.attn_weights1 = self.attn_layer1(x)
    x, self.attn_weights2 = self.attn_layer2(x)
    x, self.attn_weights3 = self.attn_layer3(x)
   

    return x


class LocalVectorNet(nn.Module):
  def __init__(self, exp_name, graph_encoder=False):
    super(LocalVectorNet, self).__init__()
    self.graph_encoder = graph_encoder
    self.exp_name = exp_name

    if graph_encoder:
      self.agent_encoder = GraphVectorEncoder(**GRAPH_AGENT_ENC)
      self.obj_encoder = GraphVectorEncoder(**GRAPH_OBJ_ENC)
      self.lane_encoder = GraphVectorEncoder(**GRAPH_LANE_ENC)
    else :
      self.agent_encoder = VectorEncoder(**AGENT_ENC)
      self.obj_encoder = VectorEncoder(**OBJ_ENC)
      self.lane_encoder = VectorEncoder(**LANE_ENC)



  def forward(self, x): # x : [agent_vectors, objects_vectors, lanes_vectors]
    agent_vectors, obj_vectors, lane_vectors = x
    agnet_vectors = agent_vectors.to(DEVICE)
    obj_vectors = obj_vectors.to(DEVICE)
    lane_vectors = lane_vectors.to(DEVICE)

    agent_encoded = self.agent_encoder(agent_vectors)
    encoded_obj_vectors = self.obj_encoder(obj_vectors)
    encoded_lane_vectors = self.lane_encoder(lane_vectors)

    if self.exp_name=='Argo-GNN-GNN':
      return agent_encoded, encoded_obj_vectors, encoded_lane_vectors
    
    agent_encoded = torch.mean(agent_encoded, axis=1)
    encoded_obj_vectors = torch.mean(encoded_obj_vectors, axis=1)
    encoded_lane_vectors = torch.mean(encoded_lane_vectors, axis=1)

    return agent_encoded, encoded_obj_vectors, encoded_lane_vectors
  
  def gnn_gnn_encoder(self, batch): 
    agent_graph_data = create_agent_graph_data(batch[0], 59)
    obj_graph_data = create_obj_graph(batch[1], 60)
    lane_graph_data = create_obj_graph(batch[2], 35)

    out = self([agent_graph_data, obj_graph_data, lane_graph_data])
    return out
  
  
  def to_trans(self, batch): 
    out = self.forward(batch[:-3])

    agent = out[0]
    agent = torch.cat([agent, torch.zeros((len(agent), 1), device=DEVICE)], 1)
    agent = agent.reshape(-1, 1, AGENT_ENC['output_dim']+1)

    obj = out[1]
    obj = torch.cat([obj, torch.ones((len(obj), 1), device=DEVICE)], 1)
    obj = obj.reshape(-1, OBJ_PAD_LEN, OBJ_ENC['output_dim']+1)
    
    lane = out[2]
    lane = torch.cat([lane, 2*torch.ones((len(lane), 1), device=DEVICE)], 1)
    lane = lane.reshape(-1, LANE_PAD_LEN, LANE_ENC['output_dim']+1)
    
    data = torch.cat([agent, obj, lane], 1)
    return data


  def to_graph_data(self, batch):
    agent_data, obj_data, lane_data, gt, n_objs, n_lanes = batch
    if self.exp_name=='Argo-GNN-GNN':
      out = self.gnn_gnn_encoder(batch)
    else : 
      out = self(batch[:-3])

    batch_data = []
    batches = []

    agent = out[0]
    agent = torch.cat([agent, torch.zeros((len(agent), 1), device=DEVICE)], 1)

    obj = out[1]
    obj = torch.cat([obj, torch.ones((len(obj), 1), device=DEVICE)], 1)

    lane = out[2]
    lane = torch.cat([lane, 2*torch.ones((len(lane), 1), device=DEVICE)], 1)


    for i in range(0, len(n_lanes)-1):

      data_raw = torch.cat([agent[i].unsqueeze(0), obj[int(n_objs[i]):int(n_objs[i+1])], lane[int(n_lanes[i]):int(n_lanes[i+1])]])
      num_nodes = len(data_raw)

      data = Data()

      data.x = data_raw

      edge_index = fc_graph(num_nodes).to(DEVICE)
      data.edge_index = edge_index

      batch_data.append(data)
      batches += [*[i]*num_nodes]

    base_data = Batch.from_data_list(batch_data)
    base_data.batch = batches
    base_data.y = gt

    return base_data
  



class GlobalEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GlobalEncoder, self).__init__()

        self.transformer_conv = TransformerConv(in_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        out = self.transformer_conv(x, edge_index, return_attention_weights=True)
        x, (edge_index, self.attention_weights) = out
        x = F.relu(x)

        x = global_mean_pool(x, data.batch)
        return x





class VectorNet(nn.Module):
  def __init__(self, exp_name):
    super(VectorNet, self).__init__()
    self.exp_name = exp_name

    if self.exp_name=='Argo-GNN-GNN':
      self.local_encoder = LocalVectorNet(self.exp_name, graph_encoder=True)
    else:
      self.local_encoder = LocalVectorNet(self.exp_name)
    
    self.local_encoder.to(DEVICE)
    
    
    if self.exp_name=='Argo-pad': 
      self.global_encoder = TransformerEncoderLayer(**GLOBAL_ENC_TRANS)
    else : 
      self.global_encoder = GlobalEncoder(**GLOBAL_ENC)
    
    self.global_encoder.to(DEVICE)
    


    self.decoder = Decoder(**DECODER)
    self.decoder.to(DEVICE)




  def forward(self, x):

    if self.exp_name=='Argo-pad': 
      encoded_vectors = self.local_encoder.to_trans(x)
      encoded_vectors, global_attention_weights = self.global_encoder(encoded_vectors)
      latent_vector = encoded_vectors.mean(axis=1)

    else :
      graph_data = self.local_encoder.to_graph_data(x)
      graphloader = GDataLoader(graph_data, batch_size=len(graph_data))
    
      latent_vector = self.global_encoder(next(iter(graphloader)))
    
    out = self.decoder(latent_vector)

    return out