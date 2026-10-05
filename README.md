# 背景修复 + 示范对齐：可回退实验快照

保存日期：2026-10-06。本仓库用于保存已有正向结果，避免后续试验覆盖原方法、参数、支持集和预测。主快照是 **L_HARD**；同时完整保存 **L_GAP** 局部背景修复版本。

核对原始机器结果后，Coffee 上数值最好的门控组合是 L_HARD，不是 L_GAP。这里的“最好”限定为本轮 Coffee 开发数据上的完整门控方法，**不代表已确认跨任务泛化或论文贡献**。无门控 E_L 的步骤内指标更高，但整体指标更低。GT 背景版本 LG=75.705 是诊断 oracle，不能作为部署方法结果。

## 1. 冻结结果

Coffee 完整 38 个测试视频，K=2，两组固定正常支持/模型 seed0、seed1 等权；native256、10fps；先组合分数，再逐视频 `median_filter(size=21, mode='nearest')`。AUC 单位是百分数。

| 方法 | 整体 AUC | GT>0 AUC | 步骤内 AUC |
|---|---:|---:|---:|
| NK 原生 AEM | 60.237 | 52.465 | 52.661 |
| LK 示范对齐 + 原生背景 | 61.337 | 53.802 | 56.496 |
| **L_HARD 示范对齐 + 固定动作性硬门控** | **65.620** | **56.088** | **57.815** |
| **L_GAP 示范对齐 + 仅修复原背景** | **64.744** | **54.029** | **55.232** |
| L_P 全帧动作性乘权 | 64.297 | 51.000 | 47.437 |
| E_L 示范对齐、无背景门控 | 60.052 | 58.611 | 62.683 |

L_HARD 相对 NK 整体 +5.383pp，相对 LK +4.283pp。L_GAP 相对 LK 整体 +3.407pp，视频 bootstrap 95%CI [1.429, 5.131]；步骤内 −1.264pp，未通过原先点估计保护规则。**保存结果不改变任何历史实验的通过/停止判定。**

Oatmeal 固定方法验证也随仓库保存：L_HARD 为 42.296 / 47.715 / 57.242，LK 为 41.738 / 44.553 / 55.218，addition-vs-normal 为 29.225。整体收益区间跨 0，整体和 addition AUC 均低于常数对照 50；不能称为泛化成功。Coffee/Oatmeal 均已经用于开发。

## 2. 方法内容及准确参数

**示范对齐。** 两条正常支持视频的非背景步骤顺序分别构成线性模板，保留重复步骤。逐帧 L2 归一化 native256 特征，对每个步骤的完整支持帧库计算精确 top5 余弦均值；logit scale=8。复用两遍 exact DP：第一遍 `(drop_base=.3, node_base=0)`，第二遍 `(-100,-200)`；按查询无标签代价选择支持模板，平局取第一条。不使用发布任务 DAG，不使用查询 GT 步骤身份。

**背景修复。** 用相同 K 条正常支持的公开步骤标签，合并 `y=1[step>0]`。视频等权的线性 logistic 探针：`mean_video(mean_frame(BCE)) + .001/2 * ||w||²`，截距不惩罚；输入 L2 归一化、float64、SciPy L-BFGS-B，maxiter=1000、ftol=1e-12、gtol=1e-8。无 class balance、标准化、正则搜索或 Platt 校准。保存的 `p` 是 sigmoid 软估计，不声称是已校准后验。

冻结 K2 AEM checkpoint，使用 `state_dict`，epoch135，不用 EMA。原动作段及分数保持；每个原背景缺口按示范路由独立划段，重新计算 AEM 分数，不与相邻同类段合并。AEM action_model 返回 `(cosine+1)/2` 的相似度，历史残差 `e_L=1-sim=(1-cosine)/2`。

设 `B=1[原生预测为正步骤]`，平滑前：

```text
LK     = B * e_L
L_HARD = 1[p >= 0.5] * e_L
L_GAP  = B * e_L + (1-B) * p * e_L
L_P    = p * e_L
```

L_HARD 只用动作性决定成员资格；L_GAP 仅补原生拒绝区域。固定 0.5 没有按测试结果搜索。这不是整个系统免训练：AEM 已经在 K 条正常支持上训练，视觉/文本编码器还使用预训练知识；支持集需要现有公开步骤标签，不新增手工标注。AEM训练曾使用发布的效果/场景图辅助资产，信息预算不能写成只有原始视频。

## 3. 直接复算历史结果：CPU 即可

仓库已经包含全部 76 份 Coffee 查询预测及相关参数、路由、标签网格、原始分数、完整对照和 bootstrap，不需要下载大权重或开启 GPU。

```bash
git clone https://github.com/sgkwzp/less-example.git
cd less-example
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install -r requirements-replay.txt
python tools/replay.py
```

此命令验证所有原始文件 SHA256，重新组合 L_HARD/L_GAP，独立计算每个种子的三项 AUC，重算 2000 次配对视频 bootstrap，并逐项比对历史机器结果。输出到 `outputs/replay.json`，不改旧证据。bootstrap seed=20261005，以 **38 个视频** 为重采样单位，两固定支持运行等权，区间不覆盖重新训练/支持集抽样的全部不确定性，不作多重比较校正。

步骤内 AUC 是每个 GT 正步骤跨完整测试视频汇总、只对含两类的步骤计算 AUC，再按这些步骤帧数加权。不能用简单平均或逐视频 AUC 平均代替。

本次本机 CPU 复算已通过，独立 AUC 和历史结果误差在浮点容差内；可见 `provenance/replay_validation.json`。76 个查询的示范对齐重生成与旧路由逐帧完全一致；重新拟合的动作性概率与原值最大差小于 2.4e-12。核验记录分别为 `provenance/alignment_validation.json`、`provenance/support_fit_validation.json`。

## 4. 大文件清单：未放入普通 Git

GitHub 普通 Git 的单文件上限为 100 MiB；本仓库不上传下列大资产。**仅 clone 仓库可以复算缓存结果，但不能重新做 AEM GPU 推理。**

| 资产 | 已核对体积 | 获取位置 |
|---|---:|---|
| Coffee K2 seed0 epoch135 checkpoint | 618,574,509 bytes，约590 MiB | 下述本地/原服务器路径 |
| Coffee K2 seed1 epoch135 checkpoint | 618,574,509 bytes，约590 MiB | 下述本地/原服务器路径 |
| 42 份 Coffee native256 特征：38 查询 + 4 支持 | 合计约179 MB | 下述本地/原服务器路径 |
| EVA02-L-14-336 CLIP safetensors | 大文件；本次未能实测体积 | [Hugging Face模型](https://huggingface.co/timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k) |

两 checkpoint + 42 特征合计 **1,415,658,074 bytes（约1.416GB）**。准确逐文件大小和 SHA256 以 [external_assets.json](provenance/external_assets.json) 为准。模型与特征的原始路径为：

```text
# 本地原工程根：E:/ky/dada_backups/fx12/
# 原服务器工程根：/hy-tmp/
ccfa-workfiles/experiments/fewshot-pmd/g1_runtime/checkpoints/aem/coffee_k2_seed0_final/epoch_135.pth.tar
ccfa-workfiles/experiments/fewshot-pmd/g1_runtime/checkpoints/aem/coffee_k2_seed1_final/epoch_135.pth.tar
code/AEM/data/egoper/coffee/vc_v_features_10fps/<video_id>.npy
```

本地两份 checkpoint 和 42 个特征在打包时全部逐文件核对历史 SHA256；原服务器路径只作原始资产定位，**本次 SSH 连接被拒绝，未确认服务器当前资产和 GPU 状态**。checkpoint/native256 特征目前没有本仓库专属的公开下载链接；它们不是 GitHub 上已备份的文件。不要删除上述本地原资产。

原生 native256 是 VideoCLIP 特征，**不能以 AEM README 的 I3D、DINO 或别的 256 维特征替代**。原始数据获取与许可见 [EgoPER](https://github.com/robert80203/EgoPER_official)。上游公开 full checkpoint 不能替代本实验 K2 checkpoint。

## 5. 重新推理：Linux + 单张4090

历史环境：Python3.11.11、torch2.5.1+cu124、OpenCLIP2.24.0、timm0.9.16、torch-geometric2.5.3，4090。`requirements-inference.txt` 是复现入口依赖，CUDA PyTorch 建议从 cu124 官方源安装；Linux 还需要 C++ 编译器/OpenMP，用于编译 NMS。推荐直接使用历史已有环境。

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-inference.txt

# ASSET_ROOT 保持 external_assets.json 中的相对目录结构。
python tools/infer_frozen.py \
  --asset-root /hy-tmp \
  --clip-weights /path/to/open_clip_model.safetensors \
  --out outputs/fresh
python tools/replay.py --predictions outputs/fresh --out outputs/fresh_metrics.json
```

入口验证所有外部资产 SHA256，从两正常支持重新拟合动作性，再用已冻结原生背景和示范片段重新计算 AEM 分数；不读查询 GT。生成新目录，禁止覆盖旧输出。重新推理的结果和历史残差最大差会逐视频保存。

若还要重生成示范对齐，而不是用已存路由：

```bash
python tools/align_support.py --asset-root /hy-tmp --out outputs/routes
python tools/infer_frozen.py \
  --asset-root /hy-tmp --clip-weights /path/to/open_clip_model.safetensors \
  --routes outputs/routes --out outputs/realigned
python tools/replay.py --predictions outputs/realigned --out outputs/realigned_metrics.json
```

`align_support.py` 只读取两条支持的步骤标注及查询特征；它沿用已冻结原生背景 mask。**这是复现历史固定路由实验的入口，不是原生分割器数值复现保证。** 原历史 H 与当前环境 C 曾有少量边界差异，所以新生成原生路由应另存、另报告，不混用旧 H 分数。原始分割/评分实现完整保存在 `code/AEM/libs/`，历史控制脚本也随仓库保存。

GPU wrapper 在本次打包中做了源码和语法核查；由于原服务器不可连接，**本次未重新完成其 GPU 运行验证**。历史原始 GPU 脚本和输出的完整核验记录已经保存。此限制不影响 CPU 缓存结果的精确复算。旧服务器如仍有 libcuda 版本问题，可沿用已验证的进程设置 `LD_LIBRARY_PATH=/hy-tmp/cuda-driver-535`，不要改系统驱动。

## 6. 文件入口与回退方式

所有历史实验文件维持原目录 `ccfa-workfiles/experiments/fewshot-pmd/`，原文件不重写：

| 内容 | 入口 |
|---|---|
| 主方法动作性、缺口构造与冻结AEM评分 | `actionness_gate.py`、`actionness_gate_gpu.py` |
| 示范对齐 | `aem_support_route.py`、`grr_validate.py`、`fast_alignment.py` |
| L_HARD完整参数/预测 | `actionness_gate_runtime/{fits,inputs,scores}/` |
| L_GAP公式与全部输出 | `actionness_local_repair.py`、`actionness_local_repair_runtime/` |
| 原生 C/NK 锚点 | `aem_cross_route_runtime/replay/C/s{0,1}_KK/` |
| Coffee完整对照和注册/核验 | `actionness-gate-{protocol,results}.md`、对应 runtime JSON |
| L_GAP完整结果及停止判定 | `actionness-local-repair-{protocol,results}.md` |
| Oatmeal跨任务结果与局限 | `hard-gate-transfer-results.md`、`hard_gate_transfer_runtime/` |
| 支持集与训练配置 | `g1_runtime/support/`、`g1_runtime/configs/aem/` |
| 所有原文件身份 | `provenance/original_files.sha256.json` |
| 外部资产身份 | `provenance/external_assets.json` |

历史 `prepare/register/routes/gpu` 脚本使用严格冻结哈希，可能引用未打包的先前诊断和服务器绝对路径；它们保存原实现与审计证据，**不应直接对已完成 runtime 重跑**。独立复现使用本 README 的 `tools/` 入口。发布标签 `coffee-k2-bg-align-v1` 固定这份结果；后续实验新建目录和 commit，禁止覆盖本快照。需要回退时使用新 checkout/worktree 从该标签开始。

## 7. 上游来源

本仓库包含本地实际使用的 AEM 与 GTG2Vid 部分源码快照及本实验扩展，逐文件 SHA256 定位具体实现。原副本不含 `.git` 元数据，不能凭空给出上游 commit SHA。上游根 README 保留作者与引用；这里不声称拥有上游代码或数据的再授权权利。

- AEM: [Procedural Mistake Detection via Action Effect Modeling](https://arxiv.org/abs/2512.03474)，Wenliang Guo、Yujiang Pu、Yu Kong。
- GTG2Vid: [ICCV 2025论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Lee_Error_Recognition_in_Procedural_Videos_using_Generalized_Task_Graph_ICCV_2025_paper.pdf)，Shih-Po Lee、Ehsan Elhamifar；仅复用其对齐代价/DP，当前示范模板不使用发布任务 DAG。
- [ActionFormer](https://github.com/happyharrycn/actionformer_release)、[Graph2Vid](https://github.com/SamsungLabs/Graph2Vid)、[EgoPER](https://github.com/robert80203/EgoPER_official)。

数据、预训练权重和上游代码继续遵循各自来源的条款。该仓库是研究复现存档，不对缺失的上游许可另作假定。
