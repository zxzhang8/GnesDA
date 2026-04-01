# 用默认参数适配 Monomer，并保持 Backbone 不变

  ## Summary

  将 monomer 兼容性从 model/GnesDA_backbone.py 移回参数层，不再依赖 backbone 内部的 patch/
  stride 防御性收敛逻辑。默认目标是 150 到 200bp 左右的 alpha satellite monomer，因此把 DNA
  相关默认超参改成对短序列安全且可训练的组合，同时保留 main.py 里的自动调参作为兜底，不修改
  backbone 结构代码。

  默认参数采用：

  - conv_layers=3
  - patch_len=16
  - stride=8

  理由是对当前已观察到的 monomer 长度范围，这组值满足卷积下采样后仍能切出合法 patch；相比现有
  默认 conv_layers=5，不会在短序列上直接触发 unfold 越界。

  ## Implementation Changes

  - 参数默认值调整
    在 main.py 的 argparse 定义里，把 --conv_layers 默认值从 5 改为 3，保留 --patch_len=16 和
    --stride=8 不变。
  - 移除对 backbone 的特殊处理依赖
    后续实现不再要求 model/GnesDA_backbone.py 内部根据 seq_len 自动收缩 patch_len/stride。
    backbone 维持论文风格的原始行为，由输入参数保证合法性。
  - 保留 main.py 的 DNA 自动调参
    tune_dna_backbone() 暂时保留，作为异常短序列或未来不同 DNA 数据的保险机制；但对 monomer
    常规运行，不应再触发或只极少触发。
  - 同步运行脚本
    更新 start_dna.sh 中显式传入的 --conv_layers 5，改为 --conv_layers 3，避免脚本覆盖新的默
    认值。
  - 同步文档
    在 README.md 的 DNA/monomer 示例中明确说明：monomer 推荐使用 conv_layers=3；如果采用默认
    参数，默认值已按 monomer 场景调整。

  ## Public Interface Changes
  - 命令行接口名称不变。
  - 实际默认行为变化：
      - --conv_layers 默认从 5 变为 3
      - --patch_len 仍为 16
      - --stride 仍为 8
  - 对用户的影响：
      - 不显式传参时，DNA monomer 训练将直接使用更适合短序列的卷积深度。
      - 若用户想复现旧配置，仍可手动传 --conv_layers 5。

  ## Test Plan

  - 静态验证
    检查 main.py、start_dna.sh、README.md 中关于 conv_layers 的默认值和示例是否一致。
  - Monomer smoke test
    用小规模 monomer FASTA 子集运行一次 --data_type dna --dist_type ed，确认：
      - 不改 backbone 也不会在 unfold 时报错
      - 训练、embedding、相关性评估完整跑通
  - 默认值行为验证
    不显式传 --conv_layers，确认程序打印出的默认值为 3。
  - 回归检查
    显式传 --conv_layers 5 时，程序仍接受该参数；如果自动调参被保留，则确认它会在必要时兜底，
    而不是静默导致非法 patch 设置。

  ## Assumptions

  - 目标数据主要是 alpha satellite monomer，长度大致在 150 到 200bp，而不是更长的通用 DNA 序
    列。
  - 当前优先级是“默认开箱即用地支持 monomer”，不是“保持与旧 DNA 默认配置完全一致”。
  - 自动调参保留只是兜底，不作为主要适配机制；主路径依赖新的默认参数组合。