import torch
import torch.nn as nn
import torch.nn.functional as F

from OneStepFormer.models.encoder import Encoder, EncoderLayer
from OneStepFormer.models.decoder import Decoder, DecoderLayer
from OneStepFormer.models.attn import AttentionLayer, FullAttention
from OneStepFormer.models.utils import PositionalEmbedding


class OneStepFormer(nn.Module):
    """
        motion trend modeling
    """

    def __init__(self, model_params):
        super(OneStepFormer, self).__init__()
        self.seq_len = model_params['seq_len']
        self.pred_len = model_params['pred_len']
        d_inp = model_params['in_dim']
        d_model = model_params['mhsa_dim']
        d_ff = model_params['mhsa_d_ff']
        n_heads = model_params['num_heads']
        dropout = model_params['dropout']
        activation = model_params['activation']

        self.enc_embedding = nn.Linear(d_inp, d_model)
        self.dec_embedding = nn.Linear(d_inp, d_model)
        self.pos_proj = PositionalEmbedding(d_model=d_model, max_len=self.seq_len + self.pred_len)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.encoder = None
        if model_params['e_layers'] > 0:
            encoder_layers = [
                EncoderLayer(
                    AttentionLayer(FullAttention(attention_dropout=dropout), d_model, n_heads),
                    d_model, d_ff, dropout, activation
                ) for _ in range(model_params['e_layers'])
            ]
            self.encoder = Encoder(encoder_layers, norm_layer=torch.nn.LayerNorm(d_model))

        decoder_layers = [
            DecoderLayer(
                AttentionLayer(FullAttention(attention_dropout=dropout), d_model, n_heads),
                AttentionLayer(FullAttention(attention_dropout=dropout), d_model, n_heads),
                d_model, d_ff, dropout, activation
            ) for _ in range(model_params['d_layers'])
        ]
        # self.decoder = Decoder(decoder_layers, norm_layer=torch.nn.LayerNorm(d_model))
        self.decoder = Decoder(decoder_layers, norm_layer=None)
        self.out_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, model_params['motion_out_dim'])
        )

    def forward(self, x_enc, x_enc_padding_mask, latest_pos):
        """
            :param x_enc: (seq_len, batch size, 4)
            :param x_enc_padding_mask: (seq_len, batch size, 1)
            :param latest_pos: for postprocess, (1, batch size, 4)
        """
        # with_pos_embed
        x_dec = torch.cat((x_enc, torch.zeros([self.pred_len] + list(x_enc.shape[1:]), device=x_enc.device)), dim=0)
        x_pos = self.pos_proj(x_dec).permute(1, 0, 2)  # (seq_len + pred_len, 1, d_model)

        x_enc = self.enc_embedding(x_enc)
        x_enc = x_enc + x_pos[:self.seq_len]
        enc_out = self.encoder(x_enc, x_enc_padding_mask)
        # x_dec: (seq_len+pred_len, batch size, 4)

        x_dec = self.dec_embedding(x_dec)
        x_dec = x_dec + x_pos
        dec_out = self.decoder(x_dec, enc_out,
                               torch.cat((x_enc_padding_mask, torch.zeros([self.pred_len, x_dec.shape[1], 1], device=x_enc.device)), dim=0),
                               x_enc_padding_mask)
        dec_out = self.out_projection(dec_out)
        pred = self.post_process(dec_out[-self.pred_len:, :, :], latest_pos)
        return pred.permute(1, 0, 2)

    def post_process(self, dec_out, latest_pos):
        pos_idx = torch.arange(1, self.pred_len+1, device=dec_out.device).reshape(-1, 1, 1).expand(-1, dec_out.shape[1], dec_out.shape[2])
        pred = dec_out * pos_idx + latest_pos
        return pred
