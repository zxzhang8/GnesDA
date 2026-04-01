from torch import nn

from model.GnesDA_backbone import GnesDA_backbone


class GnesDAModel(nn.Module):
    def __init__(self, configs, channels, max_seq_len, embed_len, d_k=None, d_v=None, attn_dropout=0., act="gelu",
                 is_mask=False, store_attn=False, pe='sincos', learn_pe=True):
        """GnesDA 外层封装。

        论文对应结构:
            输入统一表示 -> 卷积块 -> patch + Transformer -> MLP -> 最终 embedding

        参数:
            channels: 统一输入后的通道数 D
            max_seq_len: 统一后的最大长度 M
            embed_len: 总 embedding 长度，最终会被拆成 [D, embed_len // D]
        """
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
        # 当前实现里这两次 permute 会相互抵消，因此输入形状保持不变:
        #   protein: x 初始就是 [B, C, M]
        #   traj:    调用侧传入 [B, M, 2]，在 backbone 的 Pretreatment 中完成映射
        x = x.permute(0, 2, 1)  # [B, M, C] 或 [B, 2, M]

        x = x.contiguous().permute(0, 2, 1)  # [B, C, M] 或 [B, M, 2]
        x = self.model(x)  # [B, C, T]，其中 T = embed_len // C
        return x
