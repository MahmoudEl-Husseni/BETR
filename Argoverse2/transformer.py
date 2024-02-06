import torch 
import torch.nn as nn

import torch.nn.functional as F

from config import *

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=60):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        if d_model%2:
          self.encoding[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else :
          self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        return x + self.encoding[:, :x.shape[1]].to(DEVICE)
    


class Feedforward(nn.Module):
    def __init__(self, d_model, d_ff, output_dim, dropout=0.1):
        super(Feedforward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    


class MultiheadSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiheadSelfAttention, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.query = nn.Linear(input_dim, input_dim, device=DEVICE)
        self.key = nn.Linear(input_dim, input_dim, device=DEVICE)
        self.value = nn.Linear(input_dim, input_dim, device=DEVICE)
        self.fc_out = nn.Linear(input_dim, input_dim, device=DEVICE)

    def forward (self, query : torch.Tensor, key : torch.Tensor, value : torch.Tensor) -> torch.Tensor:
        '''
        query shape     -> [bs, n_tokens, d_in]
        key shape       -> [bs, n_tokens, d_in]
        value shape     -> [bs, n_tokens, d_in]
        '''
        batch_size = key.shape[0]
        query = self.query(query)   # [bs, n_tokens, d_in] -> [bs, n_tokens, d_in]
        key = self.key(key)         # [bs, n_tokens, d_in] -> [bs, n_tokens, d_in]
        value = self.value(value)   # [bs, n_tokens, d_in] -> [bs, n_tokens, d_in]

        # Split the queries, keys, and values into multiple heads
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)   # [bs, n_tokens, d_in] -> [bs, n_heads, n_tokens, d_head]
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)       # [bs, n_tokens, d_in] -> [bs, n_heads, n_tokens, d_head]
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)   # [bs, n_tokens, d_in] -> [bs, n_heads, n_tokens, d_head]

        # Scaled dot-product attention
        scaled_attention = torch.matmul(query, key.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)) # [bs, n_heads, n_tokens, n_tokens]

        attention_weights = F.softmax(scaled_attention, dim=-1) # [bs, n_heads, n_tokens, n_tokens]
        output = torch.matmul(attention_weights, value)         # [bs, n_heads, n_tokens, d_head]

        # Concatenate and linearly project the output
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.input_dim) # [bs, n_tokens, d_in]
        output = self.fc_out(output)                                                          # [bs, n_tokens, d_in] -> [bs, n_tokens, d_in]

        return output, attention_weights



class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, output_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        # Multihead Self-Attention
        self.self_attention = MultiheadSelfAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model).to(DEVICE)

        # Conversion layer
        self.convert = nn.Linear(d_model, output_dim).to(DEVICE)
        self.act = nn.ReLU().to(DEVICE)
        self.layernorm_convert = nn.LayerNorm(output_dim).to(DEVICE)


        # Feedforward

        # self.feedforward = Feedforward(d_model, d_ff, output_dim, dropout)

        self.feedforward = Feedforward(output_dim, d_ff, output_dim, dropout)
        self.norm2 = nn.LayerNorm(output_dim).to(DEVICE)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-Attention
        attention_output, att_weigths = self.self_attention(x, x, x) #  -> bs * n_heads * dim = 27 * 1 * 51
        # print(attention_output.shape)

        x = x + self.dropout(attention_output)
        x = self.norm1(x)

        # Conversion layer
        x = self.convert(x)
        x = self.act(x)
        x = self.layernorm_convert(x)


        # Feedforward
        ff_output = self.feedforward(x)
        ff_output = x + self.dropout(ff_output)
        x = self.norm2(ff_output)

        return x, att_weigths