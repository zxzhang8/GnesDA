import os
import torch
import numpy as np
from torch import nn
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model.RevIN import RevIN
from model.pretreatment import Pretreatment
from model.GnesDA_layers import positional_encoding, Transpose, get_activation_fn


class GnesDA_backbone(nn.Module):
    def __init__(self, c_in, seq_len, embed_len, patch_len=16, stride=8,
                 n_layers=6, d_model=64, n_heads=1, d_k=None, d_v=None, d_ff=256, norm=None, attn_dropout=0.,
                 dropout=0., act="gelu", is_mask=False, store_attn=False, pret_type="conv", conv_layers=6,
                 conv_channels=10, pe='sincos', learn_pe=True, padding_patch=None,
                 revin=True, affine=False, subtract_last=False, data_type="protein"):

        super().__init__()
        self.is_mask   = is_mask
        self.embed_len = embed_len
        self.do_patching = True
        self.c_in = c_in

        # Pretreatment
        self.pretreatment = Pretreatment(pret_type=pret_type, channel=c_in, M=seq_len, conv_layers=conv_layers,
                                         conv_channels=conv_channels, data_type=data_type)
        if pret_type == "conv":
            seq_len = seq_len // (2 ** conv_layers)

        # RevIn
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        self.stride        = stride
        self.patch_len     = patch_len
        self.padding_patch = padding_patch
        self.patch_num     = int((seq_len - patch_len) / stride + 1)
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1
        if not self.do_patching:
            self.patch_num = seq_len
            self.patch_len = c_in

        # Backbone
        self.backbone = GnesDAiEncoder(self.c_in, patch_num=self.patch_num, patch_len=self.patch_len, n_layers=n_layers,
                                    d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                    attn_dropout=attn_dropout, dropout=dropout, act=act, store_attn=store_attn,
                                    pe=pe, learn_pe=learn_pe)

        # MLP
        self.flat_size = d_model * self.patch_num
        self.flatten = nn.Flatten(start_dim=-2)
        self.MLP = nn.Sequential(
            nn.Linear(self.flat_size, self.flat_size),
            nn.Linear(self.flat_size, embed_len),
            nn.Linear(embed_len, embed_len),
            # nn.ReLU(),
            nn.Linear(embed_len, embed_len // c_in),
        )

    def forward(self, z):  # z: [bs x nvars x seq_len]
        z = self.pretreatment(z)

        attn_mask = None
        if self.is_mask:
            x         = torch.sum(z, dim=1)  # x: [Batch size, Input length]
            lens      = torch.count_nonzero(x, dim=1)
            lens      = ((lens - self.patch_len) / self.stride + 1).int() + (self.padding_patch == 'end')
            attn_mask = get_attn_mask(self.patch_num, lens).to("cuda")

        # norm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0, 2, 1)

        # do patching
        if self.do_patching:
            if self.padding_patch == 'end':
                z = self.padding_patch_layer(z)
            z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # z: [bs x nvars x patch_num x patch_len]
            z = torch.reshape(z, (z.shape[0] * z.shape[1], z.shape[2], z.shape[3]))  # z: [bs*nvars x patch_num x patch_len]
        else:
            z = z.transpose(1, 2)

        # model
        z = self.backbone(z, attn_mask)  # z: [bs*nvars x patch_num x d_model]
        if self.do_patching:
            z = torch.reshape(z, (-1, self.c_in, z.shape[-2], z.shape[-1]))
        z = self.flatten(z)
        z = self.MLP(z)

        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0, 2, 1)

        return z


def get_attn_mask(max_len, lens):
    batch_size = len(lens)
    k = torch.zeros((len(lens), max_len))  # k: [bs x max_len]
    for i, l in enumerate(lens):
        k[i, :l] = 1
    attn_mask = k.data.eq(0).unsqueeze(1)  # [batch_size, 1, max_len], True is masked

    return attn_mask.expand(batch_size, max_len, max_len)


class GnesDAiEncoder(nn.Module):
    def __init__(self, c_in, patch_num, patch_len, n_layers=6, d_model=64, n_heads=1, d_k=None, d_v=None, d_ff=256,
                 norm=None, attn_dropout=0., dropout=0., act="gelu", store_attn=False, pe='sincos', learn_pe=True):
        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len
        self.d_model   = d_model

        # Input encoding
        self.W_P = nn.Linear(patch_len, d_model)

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, patch_num, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = GnesDAEncoder(d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, activation=act, n_layers=n_layers, store_attn=store_attn)

    def forward(self, x, attn_mask=None):  # x: [bs*nvars x patch_num x patch_len]
        # Input encoding
        x = self.W_P(x)  # x: [bs*nvars x patch_num x d_model]
        u = self.dropout(x + self.W_pos)  # u: [bs*nvars x patch_num x d_model]

        # Encoder
        z = self.encoder(u, attn_mask=attn_mask)  # z: [bs*nvars x patch_num x d_model]
        return z


# Cell
class GnesDAEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=None, norm=None, attn_dropout=0., dropout=0.,
                 activation='gelu', n_layers=6, store_attn=False):
        super().__init__()

        self.attn_layers = nn.ModuleList(
            [GnesDAEncoderLayer(d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                             dropout=dropout, activation=activation, store_attn=store_attn) for i in range(n_layers)
             ]
            )

    def forward(self, src, attn_mask=None):
        output = src

        for attn_layer in self.attn_layers:
            output = attn_layer(output, attn_mask=attn_mask)

        return output


class GnesDAEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm=None, attn_dropout=0., dropout=0., bias=True, activation="gelu"):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if norm is not None:
            if "batch" in norm.lower():
                self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
            else:
                self.norm_attn = nn.LayerNorm(d_model)

        # Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if norm is not None:
            if "batch" in norm.lower():
                self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
            else:
                self.norm_ffn = nn.LayerNorm(d_model)

        self.norm = norm

    def forward(self, src, attn_mask=None):
        # Attention
        src2 = self.self_attn(src, src, src, attn_mask=attn_mask)

        # Add & Norm
        src = src + self.dropout_attn(src2)
        if self.norm is not None:
            src = self.norm_attn(src)

        # Feed-Forward
        src2 = self.ff(src)

        # Add & Norm
        src = src + self.dropout_ffn(src2)
        if self.norm is not None:
            src = self.norm_ffn(src)

        return src


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, attn_dropout=0., proj_dropout=0.):
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=True)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=True)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=True)

        # Scaled Dot-Product Attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model),
                                    nn.Dropout(proj_dropout)
                                    )

    def forward(self, Q, K=None, V=None, attn_mask=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)      # q_s: [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)  # k_s: [bs x n_heads x d_k x q_len]
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)      # v_s: [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention
        output = self.sdp_attn(q_s, k_s, v_s, attn_mask=attn_mask)

        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        return output


class _ScaledDotProductAttention(nn.Module):

    def __init__(self, d_model, n_heads, attn_dropout=0.):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.n_heads      = n_heads
        head_dim          = d_model // n_heads
        self.scale        = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=False)

    def forward(self, q, k, v, attn_mask=None):
        attn_scores = torch.matmul(q, k) * self.scale  # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Attention mask
        if attn_mask is not None:
            channels    = q.shape[0] // attn_mask.shape[0]
            attn_scores = attn_scores.view(-1, channels, attn_scores.shape[1], attn_scores.shape[2], attn_scores.shape[3])  # attn_scores: [bs x channels x n_heads x max_q_len x q_len]
            attn_mask   = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)  # attn_mask: [bs x n_heads x seq_len x seq_len]
            attn_mask   = attn_mask.unsqueeze(1).repeat(1, channels, 1, 1, 1)  # attn_mask: [bs x channels x n_heads x seq_len x seq_len]
            attn_scores.masked_fill_(attn_mask, -np.inf)
            attn_scores = attn_scores.view(-1, attn_scores.shape[-3], attn_scores.shape[-2], attn_scores.shape[-1])

        attn_weights = F.softmax(attn_scores, dim=-1)  # attn_weights   : [bs x n_heads x max_q_len x q_len]
        # for i in range(0, attn_weights.shape[0], channels):
        #     visualize(attn_weights[i, 0].detach().cpu().numpy(), i)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, v)  # output: [bs x n_heads x max_q_len x d_v]

        return output

def visualize(attn_scores, batch_num):
    non_zero_cols = np.any(attn_scores != 0, axis=0)
    attn_scores = attn_scores[non_zero_cols][:, non_zero_cols]

    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_scores, annot=True, fmt=".1f", cmap='viridis')

    plt.title("Attention Scores Heatmap")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.savefig("heatmap.png")
    
    output_path = os.path.join("heatmaps", f"heatmap_batch_{batch_num}.png")
    plt.savefig(output_path)

    plt.close()

