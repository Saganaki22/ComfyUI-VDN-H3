# ComfyUI-VDN-H3 — 面向 MiniMax-H3 的 VDN-H3(Video Delta Net)混合注意力

<img width="1039" height="505" alt="image" src="https://github.com/user-attachments/assets/ab4c1691-bff5-46fe-8b3e-635429b0700f" />

**[English](README.md)** | 中文

将 Video Delta Net 混合注意力以原生 ComfyUI 节点的形式带入 MiniMax-H3:邻近帧
保留精确 softmax 注意力,远距离时序上下文交给检查点中的 **Video Delta
Attention** 线性分支,把平方级的长距离注意力替换为常数成本的循环状态。

参考实现:[OpenVDN/vdn-minimax-h3](https://github.com/OpenVDN/vdn-minimax-h3)
(Apache-2.0)。权重:[OpenVDN/vdn-minimax-h3](https://huggingface.co/OpenVDN/vdn-minimax-h3)
(MiniMax H3 社区许可证 —— **使用前请阅读**,该许可证排除部分地区)。

本包是**移植而非分叉**:在 ComfyUI 原生 MiniMax-H3 模型上以运行时模型补丁的
方式复现官方混合注意力数学,不修改任何 ComfyUI 核心文件。

**这个仓库存在的意义(以及它不是什么)。** 官方 VDN-H3 发布版面向数据中心技术栈:8× B200 GPU 的 Ulysses 序列并行,以及仅支持 Hopper 和数据中心级 Blackwell 的 FlashAttention-4 内核 —— 消费级 Blackwell(sm_120)不受支持,也没有 Windows 构建。上游还使用了 FP8 线性层和定制融合 Triton 内核;本移植用可在任何 ComfyUI 环境运行的纯 PyTorch 等价实现替代了这些。

你能得到:相同的发布检查点与相同的架构 —— 窗口 softmax + Video Delta Attention 线性分支,并对官方实现做了单元测试验证 —— 零新增依赖。8 步蒸馏模型、相对稠密 H3 近乎无损的质量,以及随片段长度线性(而非平方)增长的注意力成本 —— 视频越长,收益越大。

你得不到:头条数字。官方 74.5 倍来自 8 卡并行 + FA4 + FP8 + 8 步蒸馏的组合;上游自己的单卡实测为 50 步约 2.6 倍,而本移植的通用内核略低于此(RTX 5090、1280×736 / 145 帧实测约 17 秒/it —— 见 Benchmarks.md)。想在自己的硬件上试验这套架构,这就是为你准备的;想要实时流式生成的数字,那需要他们的 B200 集群。

**硬件现实检查。** 这不是运行 MiniMax-H3 最快或最轻量的方式 —— 这是一个实验性的 PyTorch 移植,用相似的数学复现相似的结果。本节点在每个 transformer 块、每个采样步上都要额外运行一个线性分支网络,所需算力与显存远高于 int8 融合注意力(comfy-kitchen)、SageAttention、SOL 或 SLA —— 用那些方法,同样的显卡大约能跑两倍的分辨率与时长。换来的是:片段越长、分辨率越高,收益越大 —— VDN 的注意力开销随时长线性增长而非平方增长 —— 前提是你的显存喂得饱它。上游方法面向 8× B200 数据中心 GPU 集群设计,并非消费级硬件。**显存或内存紧张的话,不建议使用本仓库/模型/方法。**

| CK、Sol-attn、res_multi / simple —— 20 步，1280x736，3:05 | LightXv2 4-Step Turbo v1.1、CK、Sol-attn、er_sde / beta —— 8 步，1280x736，1:24 |
|:---:|:---:|
| <video src="https://github.com/user-attachments/assets/7120657d-af61-4414-b621-53b39208ffe0" controls></video> | <video src="https://github.com/user-attachments/assets/b0373566-fc78-4616-b591-13462c4b50e6" controls></video> |

| VDN-H3 Turbo、er_sde / beta —— 8 步，1280x736，2:04 | VDN-H3 高级 fast_kernels Turbo、er_sde / beta —— 8 步，1280x736，1:13 |
|:---:|:---:|
| <video src="https://github.com/user-attachments/assets/89cc7155-ca89-459e-9996-5b5f6bfcd284" controls></video> | <video src="https://github.com/user-attachments/assets/5cc9906e-acec-4c61-a3b9-17c79153945b" controls></video> |

<details>
<summary><strong>VDN-H3 bf16 与 INT8 ConvRot —— A/B 视频（相同种子与设置）—— 点击展开</strong></summary>

| VDN-H3 bf16 stage、er_sde / beta —— 8 步，1280x736，1:51 | VDN-H3 INT8 ConvRot stage、er_sde / beta —— 8 步，1280x736，1:35 |
|:---:|:---:|
| <video src="https://github.com/Saganaki22/ComfyUI-VDN-H3/releases/download/exp-int8-media/ab_bf16_8step_er_sde_beta.mp4" controls loop></video> | <video src="https://github.com/Saganaki22/ComfyUI-VDN-H3/releases/download/exp-int8-media/ab_int8convrot_8step_er_sde_beta.mp4" controls loop></video> |

两侧种子与设置完全相同（merge、`cache_gpu`），仅 stage 不同。INT8 stage 的
分支矩阵乘法快 2.7 倍，此单次 A/B 端到端快约 1.2 倍，输出一致。

</details>

**Ref2V 示例（INT8 ConvRot stage）** —— ref2va 基座，8 步，er_sde / beta，
768x768：

https://github.com/user-attachments/assets/65fd49e1-a4a3-4e28-9f3d-9dc8337354a7

### 相同种子

`981445682258077`

## 安装

1. 克隆到 `ComfyUI/custom_nodes/` 并重启 ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Saganaki22/ComfyUI-VDN-H3
```

2. 将需要的 VDN 检查点下载到 `ComfyUI/models/vdn/`:

```bash
hf download OpenVDN/vdn-minimax-h3 --include "stage-dmd-step-250/*" --local-dir <ComfyUI>/models/vdn
```

请保持发布目录结构不变(`model_spec.json`、`linear_branch/`、`adapters/`)。
磁盘上不做任何转换 —— 节点在内存中把 diffusers 格式的张量键映射到 ComfyUI
的模块路径。

**无需安装任何新 Python 依赖。** 节点以 ComfyUI 自带的 PyTorch(torch +
safetensors)运行官方数学,不需要 Triton、flash-attn-4、CUDA 编译或
`pip install`。

## 节点

**Apply VDN-H3 (MiniMax-H3 Hybrid Attention)** —— `MODEL -> MODEL`

| 输入 | 含义 |
|---|---|
| `vdn_checkpoint` | `models/vdn` 下的某个 stage 目录 |
| `apply_turbo_adapter` | 开 = 官方 **8 步** 模型(采样器用 8 步);关 = **50 步** 模型(约 50 步) |
| `strength` | 适配器强度,1.0 即发布模型 |
| `lora_mode` | **`merge`**(默认;适配器合并进权重——精确复现验证过的模型)/ `bypass`(运行时注入) |

> **`lora_mode` —— 请用 `merge`,8 步 DMD 检查点(`stage-dmd-*`)尤其必须。**
> 在剪枝 int8 基座上实测:bypass 应用的是同样的适配器,但每个模块的增量以
> bf16 舍入噪声的形式叠加,而不是烧进权重。前 34 个块与 merge 逐位一致;
> 深层块(34+)会把这部分噪声放大到特征幅度的约 10%,8 步模型的 bypass
> 渲染全部出现颗粒感/劣化。同样大小的连贯扰动(强度 1.016)渲染干净——
> 问题特定于脱离流形的舍入噪声,而非增量数学本身。stage-dmd-* 必须用
> merge;非 DMD 检查点仍可使用 bypass。
| `branch_weights` | `stream`(权重按块逐步从磁盘直读进 GPU——不在内存中额外驻留;小显存安全)/ `cache_gpu`(常驻显存,更快,需预留约 4.3 GB) |
| `attention_backend` | `grouped`(默认;每个窗口组一次稠密 SDPA)/ `flex`(单个编译的 FlexAttention 内核;可选,见 Benchmarks.md) |
| `verbose` | 输出已应用的适配器和每次前向的布局日志 |

把它接在 MiniMax-H3 加载器和采样器之间即可;条件、LoRA、采样器、VAE 解码
和视频/音频输出节点都不需要改动。示例工作流:`example_workflows/vdn_h3_t2v_8step.json`。

**Apply VDN-H3 Advanced** —— 包含基础节点全部功能,另加面向实验者的:

| 输入 | 含义 |
|---|---|
| `stage_b_strength` / `turbo_strength` | 两套适配器各自独立强度(基础节点是单一全局强度) |
| `window_radius`、`window_chunk` | 偏离训练窗口 c=5 r=1(消融实验) |
| `anchor_frames` | `both` / `columns` / `rows` / `none`(训练值 `both`) |
| `text_state` | 初始化时把提示词写入线性分支状态(训练值:开) |
| `linear_branch` | 关 = 仅窗口消融(调试;长片段将失去全部长距离上下文) |
| `fast_kernels` | torch.compile 把分支热点(RMSNorm+门控尾声、状态收集、帧主序 q 存储、双向扫描为单次 CUDA-graph 重放)融合为单内核(数学相同;编译失败自动回退 eager)。**已知在 torch 2.10 上对 8 步 DMD stage(`stage-dmd-*`)产生可见漂移** —— 融合内核的 bf16 舍入误差被蒸馏采样器放大。仅限消融实验;最终渲染请关闭(节点会打印警告) |

消融输入偏离检查点训练规格时会在控制台警告;全部默认值精确复现发布模型。

## 注意力后端与叠加

VDN 的窗口 softmax 始终使用精确 SDPA —— 经由 ComfyUI 的后端优先级链分发
(flash / cuDNN / mem-efficient),但绝不经过量化后端:让窗口走 sage/kitchen
int8 会明显降低输出质量,而发布模型验证的是精确的局部注意力。后端 override
补丁(SageAttention、kitchen-int8、KJNodes)仍作用于基座模型自身的注意力
(文本精炼器,以及极短片段的稠密回退)。线性分支不经过 softmax 内核,不受
后端补丁影响。

**请勿将 "MiniMax H3 Scheduled Sol Attention" 补丁与本节点叠加。** 它替换的
`blocks.*.attn.forward` 与 VDN 是同一路径 —— 凡由 SOL 处理的调用,VDN 的线性
分支都会被跳过,此时运行的已不再是 VDN-H3(VDN 的 LoRA 被用在了未经其训练的
注意力上)。纯 H3 跑 SOL;VDN 就跑 VDN。SOL 的 FFN 分块节点与通用注意力 override
则可以叠加。

## 所需模型

| 组件 | 文件 | 来源 | 放置于 |
|---|---|---|---|
| 基座扩散模型 | `minimax_h3_fl2va_int8_convrot.safetensors`(torch cu130 推荐;仅当无法使用时才选 `fp8_scaled` 变体) | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | `models/diffusion_models` |
| 文本编码器 | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | `models/text_encoders` |
| 视频 VAE | `minimax_h3_video_vae_fp16.safetensors` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | `models/vae` |
| 音频 VAE | `minimax_h3_audio_vae_fp32.safetensors` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | `models/vae` |
| VDN 分支 + 适配器 | `stage-dmd-step-250/`(8 步)和/或 `stage-b-step-2000/`(50 步) | [OpenVDN/vdn-minimax-h3](https://huggingface.co/OpenVDN/vdn-minimax-h3) | `models/vdn` |

VDN 发布版**不包含基座权重** —— 只有分支与 LoRA 适配器,运行时应用到你加载
的任意 MiniMax-H3 基座上。HF 仓库里 72 GB 的 diffusers 基座(`h3-base/`)
**不需要**。

**已在 `fl2v`(fl2va)与 `ref2v`(ref2va)两种 MiniMax-H3 基座模型上测试,均可正常工作。**

下载你想要的 VDN 检查点 stage 到 `ComfyUI/models/vdn/`:

```bash
hf download OpenVDN/vdn-minimax-h3 --include "stage-dmd-step-250/*" --local-dir <ComfyUI>/models/vdn
```

或者使用 8 步 stage 的预量化 **INT8 ConvRot** 版本 —
[drbaph/vdn-minimax-h3-int8-convrot-comfyui](https://huggingface.co/drbaph/vdn-minimax-h3-int8-convrot-comfyui)
(输出一致,分支 4.3 -> 2.2 GB,
加载峰值显存低约 4.7 GB,需要 v1.3.0+):

```bash
hf download drbaph/vdn-minimax-h3-int8-convrot-comfyui --local-dir <ComfyUI>/models/vdn/vdn-minimax-h3-int8-convrot-comfyui
```

文件夹名即为 `vdn_checkpoint` 中的选项。也可以用 `tools/quantize_vdn_branch_int8.py` 自行量化任意 stage。

8 步模型的 `turbo` 适配器**替代**(而非叠加)社区版 MiniMax-H3 turbo LoRA
—— 两者不要同时启用。

## 官方行为与本移植的差异

**与官方实现一致**(由 `tests/` 中的单元测试对照参考数学验证):chunk 对齐的
softmax 窗口与锚点帧(发布规格 `radius=1, chunk=5, anchor_frames=both`)、
`vdn_solve` delta rule、带 alpha 桥接与文本状态的双向帧扫描、K/V 短卷积、
输出门控,以及两套 LoRA 适配器。

**ComfyUI 特有适配:**

- 窗口 softmax 默认按 chunk 分组、每组一次稠密 SDPA,而非官方的 block-sparse
  FlexAttention。分区与数学完全一致;无需 Triton 或 torch.compile。**已内置**
  FlexAttention + BlockMask 路径(经 `attention_backend: flex` 启用),在
  triton-windows 上编译运行正常 —— RTX 5090、34.5k tokens 下与 grouped 实测
  持平(见 Benchmarks.md),故 grouped 仍为默认。官方 FA4 后端更快,但需要
  Linux + 数据中心级 Blackwell。
- 官方 Triton/编译融合点(时序卷积、RMSNorm 尾声、gather)默认在此为 eager
  实现;高级节点的 `fast_kernels` 会把尾声、状态收集与帧主序 q 存储
  torch.compile 为单内核(数学相同,失败自动回退 eager)。扫描循环的内核
  启动开销仍是下一个优化目标(torch.compile CUDA graph)。
- LoRA 通过 ComfyUI 的 bypass/merge 机制应用(int8 融合的 `fc2` 自动走 merge;
  剪枝基座获得 e-grid adaln 重注入)。
- 打包序列几何直接读取 ComfyUI 自带的 `PackedLayout`,各条件变体
  (t2va / fl2va / ref2va)保持可用;VDN 训练只覆盖过 t2va 风格布局。

## 显卡 / 平台

- **Windows + NVIDIA**:主要目标平台,已测试(RTX 5090,torch 2.10+cu130)。
- **Linux + NVIDIA**:应可同样工作(纯 PyTorch)。
- 本移植仅支持单卡。官方 Ulysses 八卡路径未实现(那是并行方式而非算法)。
- AMD/Intel/CPU:未测试;eager PyTorch 意味着能跑但很慢。delta rule 的
  Cholesky 需要批量求解后端 —— CPU 可用于小规模测试。

## 显存与性能

显存主要由基座模型决定;VDN 增加约 4.3 GB 分支权重(`stream` 模式下按块流动,
工作集增量约为一个块的 ~86 MB,外加注意力内部约 `2 x seq_len x 7168 x 2` 字节
的临时 q/k 副本)。

RTX 5090 实测(int8 convrot 基座,`stream` 模式,sage2 补丁):1280x736、
145 帧、8 步、euler/simple、seed 42,约 17 秒/it(采样约 2:15),含音频。
`grouped` 与 `flex` 注意力后端在 34.5k tokens 下实测持平 —— 该长度下 grouped
路径每块每步仅约 6 次稠密 SDPA 调用,flex 的融合尚无收益,故 grouped 仍为
默认。官方报告参考:单张 B200 上稠密 50 步模型 13.95 分钟,优化后的 VDN-H3
为 5.34 分钟(仅混合架构约 2.6 倍);头条 74.5 倍来自 8xB200 并行 + 8 步蒸馏 +
fp8 线性层 + FA4/flex 内核的组合。本移植的单卡收益应对标约 2.6 倍的架构性数字,
具体随窗口所用的注意力后端变化。完整测量数据与验证状态见
[Benchmarks.md](Benchmarks.md)。

## 故障排查

- **`VDN checkpoint ... not found`** —— stage 目录须位于 `models/vdn/` 下,
  且包含 `linear_branch/model.safetensors` 与 `model_spec.json`。
- **"checkpoint has N blocks but the loaded model has M"** —— VDN stage 与
  加载的基座不匹配(例如 50 块的 stage 用在不同深度的模型上)。请加载匹配的
  MiniMax-H3 基座。
- **"This MODEL already has VDN-H3 applied"** —— 该节点只能串接一次。
- **`aimdo memory compile error`,或采样中途进程直接崩溃(2026-09-04 之后
  的 comfy,即引入 "Comfy Compiler" 的版本)** —— comfy 新增的模型编译器 /
  malloc-graph 内存规划器无法处理 VDN 改写后的 MiniMax-H3 前向(comfy 提交
  `804eb551`,同批 `5c23fb7b` 还移除了一处 MiniMax 内存规避)。节点会自动处理:
  应用 VDN 时临时关闭 comfy 编译器,VDN 模型被卸载时立刻恢复,非 VDN 工作流
  不受影响(控制台有对应日志)。手动等效方案:启动参数加
  `--disable-comfy-compiler`,或使用 2026-09-04 之前的 comfy 版本。
- **OOM** —— 用 `branch_weights: stream`(默认)、`lora_mode: merge`、更短的
  片段或更小的分辨率。**中途取消**:VDN 会在取消时清掉自己的 GPU 缓存,让重跑
  从干净状态开始;如果是基座模型因显存压力被挤到内存,重跑前手动释放一次
  (Manager 的 Free、Unload 节点或 `POST /free`)——那部分驻留属于 comfy,不归本节点管。
- **8 步下动作异常** —— 确认 8 步配 `apply_turbo_adapter` 开,或约 50 步配关;
  两种步数混用会降低质量。
- **能出片但像纯模型** —— 打开 `verbose`,在控制台找 `[vdn] layout:`;当片段
  的潜在帧数 ≤ 15 时窗口已覆盖全部,VDN 会正确地回退到稠密注意力。

## 许可证与引用

本移植采用 Apache-2.0(见 LICENSE)。VDN-H3 架构、训练与检查点来自
[OpenVDN](https://github.com/OpenVDN/vdn-minimax-h3)(Apache-2.0);MiniMax-H3
权重遵循 MiniMax H3 社区许可证。使用 VDN-H3 请引用原作者:

```bibtex
@misc{xi2026videodeltanet,
  title  = {VideoDeltaNet on MiniMax H3},
  author = {Haocheng Xi and Yiming Xie and Hexu Zhao and Yiwen Zhang and Michael Liu and Thomas Creavin and Kurt Keutzer and Xiuyu Li and Zhaoyang Lv and Chenfeng Xu and Haiwen Feng},
  year   = {2026},
  url    = {https://openvdn.github.io/}
}
```

