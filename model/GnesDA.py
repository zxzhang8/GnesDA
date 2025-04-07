from torch import nn

from model.GnesDA_backbone import GnesDA_backbone


class GnesDAModel(nn.Module):
    def __init__(self, configs, channels, max_seq_len, embed_len, d_k=None, d_v=None, attn_dropout=0., act="gelu",
                 is_mask=False, store_attn=False, pe='sincos', learn_pe=True):
        super().__init__()

        # load parameters
        norm          = configs.norm
        data_type     = configs.data_type
        n_layers      = configs.e_layers
        conv_layers   = configs.conv_layers
        conv_channels = configs.conv_channels
        n_heads       = configs.n_heads
        d_model       = configs.d_model
        d_ff          = configs.d_ff
        dropout       = configs.dropout

        patch_len     = configs.patch_len
        stride        = configs.stride
        padding_patch = configs.padding_patch

        revin         = configs.revin
        affine        = configs.affine
        subtract_last = configs.subtract_last

        # model
        self.model = GnesDA_backbone(c_in=channels, seq_len=max_seq_len, embed_len=embed_len,
                                       patch_len=patch_len, stride=stride,
                                       n_layers=n_layers, d_model=d_model,
                                       n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                       attn_dropout=attn_dropout,
                                       dropout=dropout, act=act,
                                       is_mask=is_mask, conv_layers=conv_layers,
                                       store_attn=store_attn, conv_channels=conv_channels,
                                       pe=pe, learn_pe=learn_pe,
                                       padding_patch=padding_patch,
                                       revin=revin, affine=affine,
                                       subtract_last=subtract_last, data_type=data_type)

    def forward(self, x):  # x: [Batch, Channel, Input length]
        x = x.permute(0, 2, 1)  # x: [Batch, Input length, Channel]

        x = x.contiguous().permute(0, 2, 1)  # x: [Batch, Channel, Input length]
        x = self.model(x)  # x:[Batch, embed_dim]
        return x
