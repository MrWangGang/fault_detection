# ⚙️ 基于时频图与 LoRA 技术的旋转机械故障诊断项目 (TD-LoRA-FD)

## 简介 (Introduction)

本项目专注于旋转机械的智能故障诊断，通过将原始振动信号转换为**时频图（STFT 谱图）**，并利用 **ResNet** 等深度学习模型进行分类。项目的核心亮点在于其支持**多源数据处理（CWRU `.mat` 和工业级 `.tdms`）**，以及在目标领域（Target Domain）引入 **LoRA (Low-Rank Adaptation)** 参数高效微调技术，以解决迁移学习或领域自适应中的数据稀疏问题。

## 📁 文件结构与功能

本项目包含数据预处理、可视化报告工具和多域模型训练脚本：

| 文件夹/文件 | 描述 | 关键功能点 | 关键技术/数据类型 |
| :--- | :--- | :--- | :--- |
| **datasets/** | 存放原始信号数据，结构应按类别划分。 | N/A | `.mat` (CWRU), `.tdms` (Origin/Target) |
| **data/** | 存放预处理后生成的 224x224 STFT 图像。 | N/A | `.png` 图像文件 |
| `convert_cwru.py` | CWRU 数据集（.mat 文件）预处理脚本。 | 读取 `.mat` 文件，提取信号通道，生成 STFT 图像。 | CWRU, MAT 文件 |
| `convert_origin.py` | 原始域数据（.tdms 文件）预处理脚本。 | 固定读取 `数据/RZ` 通道，生成 STFT 图像。 | Origin Domain, TDMS 文件 |
| `convert_target.py` | 目标域数据（.tdms 文件）预处理脚本。 | 固定读取 `数据/RZ` 通道，生成 STFT 图像。 | Target Domain, TDMS 文件 |
| `pre_*.py` | 三个域的数据可视化报告脚本。 | 针对每个类别，生成包含波形图和 STFT 谱图的 PDF 报告，用于数据质量检查。 | 信号可视化, PDF 报告 |
| `Reporter.py` | 核心报告和可解释性工具模块。 | 绘制训练历史、混淆矩阵、t-SNE 降维、Grad-CAM 和 LIME 可解释性分析。 | 模型可解释性 |
| `train_cwru.py` | CWRU 数据集全量训练脚本。 | 使用 ResNet (如 `resnet50`) 模型进行多分类训练 (10 类别)。 | ResNet, 10 Classes |
| `train_origin.py` | 原始域数据集全量训练脚本。 | 使用 ResNet (如 `resnet50`) 模型进行多分类训练 (5 类别)。 | ResNet, 5 Classes |
| `train_target.py` | **目标域 LoRA 微调脚本**。 | 引入 `LoRA_Wrapper` 进行高效微调，支持迁移学习。 | ResNet-LoRA, PEFT |

## ✨ 核心技术亮点

### 1. 统一的时频图转换 (STFT)

所有原始信号数据均被统一转换为标准的 224x224 像素 STFT 谱图，作为 CNN 模型的输入。

| 参数 | 值 | 说明 |
| :--- | :--- | :--- |
| 窗口长度 (`WINDOW_LENGTH`) | `1024` | FFT 窗口的大小（时域切片长度） |
| 步长 (`STEP_SIZE`) | `256` | 窗口的滑动步长（重叠度高） |
| NFFT 大小 (`NFFT_SIZE`) | `128` | 决定了 STFT 谱图的频率分辨率 |
| 目标图像尺寸 | `224x224` | 标准化图像尺寸，适配主流预训练模型 |

### 2. LoRA 参数高效微调 (PEFT)

`train_target.py` 集成了 LoRA 技术，专为目标域微调设计：

* **LoRA 封装 (`LoRA_Wrapper`)**：通过 `register_buffer` 机制，确保原始权重 $W_0$ 被冻结，不计入可训练参数。
* **低秩分解**：仅训练低秩分解矩阵 $A$ 和 $B$ (`self.lora_A`, `self.lora_B`)，大幅减少了微调所需的参数量和计算资源。
* **适用性**：支持对 `nn.Linear` 和 `nn.Conv2d` (1x1) 模块进行 LoRA 注入。

### 3. 全面的模型报告与可解释性

`Reporter.py` 模块提供了强大的模型分析功能：

* **特征可视化**：支持 **t-SNE** 降维可视化，用于观察模型提取的特征在二维空间中的聚类效果。
* **可解释性分析**：集成 **Grad-CAM** 和 **LIME**，能够定位模型在 STFT 图像中关注的关键区域，增强诊断结果的可信度。

<img width="1000" height="350" alt="F0043_0002_CAM_True_F4_Pred_F4_Combined" src="https://github.com/user-attachments/assets/ead0ce01-3397-4386-9942-083d3a268b91" />
<img width="650" height="350" alt="F0043_0002_LIME_True_F4_Pred_F4_Combined" src="https://github.com/user-attachments/assets/836f07e1-422f-4533-8b97-0cbd3a092332" />
<img width="800" height="700" alt="seresnext50_32x4d_confusion_matrix" src="https://github.com/user-attachments/assets/664b410d-a93f-4a4f-ac0e-e1795a036cf7" />
<img width="1800" height="500" alt="seresnext50_32x4d_metrics_curves" src="https://github.com/user-attachments/assets/160f4aec-9ce3-4d77-a4f1-b5540c8ff25e" />
<img width="1500" height="1200" alt="seresnext50_32x4d_tsne_visualization" src="https://github.com/user-attachments/assets/f06c4655-b5d0-45f0-a463-289b87b4a6bf" />




