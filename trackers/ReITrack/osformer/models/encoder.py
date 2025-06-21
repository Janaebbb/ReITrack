import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, enc_layers, norm_layer=None):
        super().__init__()
        self.enc_layers = nn.ModuleList(enc_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [L, B, D]
        for enc_layer in self.enc_layers:
            x, attn = enc_layer(x, attn_mask=attn_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout=0, activation="relu"):
        super(EncoderLayer, self).__init__()
        self.attention = attention
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        """
            :param x: (seq_len, batch_size, d_model)
            :param attn_mask: (seq_len, batch_size, 1)
        """
        x, attn = self.attention(x, x, x, attn_mask)
        x = x + self.dropout(x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x+y), attn
