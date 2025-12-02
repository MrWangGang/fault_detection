import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, f1_score, accuracy_score
from collections import Counter
import numpy as np
import os
import re
import timm
from tqdm import tqdm
import math

try:
    from Reporter import plot_history, plot_confusion_matrix, extract_and_plot_tsne, \
        generate_grad_cam_images, generate_lime_analysis
except ImportError:
    raise ImportError("致命错误：无法导入 Reporter.py 中的报告函数。请确保文件存在且文件名正确。")


# ==============================================================================
# --- 新增 LoRA 核心组件：通用 LoRA 封装层 ---
# 【W_0 使用 register_buffer 解决冻结失败问题】
# ==============================================================================

class LoRA_Wrapper(nn.Module):
    """
    通用 LoRA 封装层，支持 nn.Linear 和 nn.Conv2d (1x1)。
    使用 register_buffer 确保原始权重 W_0 不被意外解冻或计入可训练参数。
    """
    def __init__(self, original_module, r, alpha):
        super().__init__()

        self.r = r
        self.alpha = alpha

        self.is_linear = isinstance(original_module, nn.Linear)

        # --- 关键修正：将 W_0 的权重和偏置作为不可训练的缓冲区保存 ---
        if hasattr(original_module, 'weight'):
            self.register_buffer('W_0_weight', original_module.weight.data.clone())
        else:
            self.W_0_weight = None

        if hasattr(original_module, 'bias') and original_module.bias is not None:
            self.register_buffer('W_0_bias', original_module.bias.data.clone())
        else:
            self.W_0_bias = None

        # 记录 W_0 的其他关键属性
        self.stride = original_module.stride if hasattr(original_module, 'stride') else 1
        self.padding = original_module.padding if hasattr(original_module, 'padding') else 0
        self.groups = original_module.groups if hasattr(original_module, 'groups') else 1

        # --- LoRA A/B 矩阵的创建 (它们是 LoRA_Wrapper 的可训练参数) ---
        if self.is_linear:
            in_features = original_module.in_features
            out_features = original_module.out_features

            # LoRA A: (r, in)
            self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
            # LoRA B: (out, r)
            self.lora_B = nn.Parameter(torch.zeros((out_features, r)))

            # 初始化 A 矩阵
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        else: # nn.Conv2d (1x1) 类型的 LoRA
            in_channels = original_module.in_channels
            out_channels = original_module.out_channels

            # 使用 nn.Conv2d 作为 LoRA 矩阵 A 和 B
            self.lora_A = nn.Conv2d(in_channels, r, kernel_size=1, bias=False)
            self.lora_B = nn.Conv2d(r, out_channels, kernel_size=1, bias=False)

            # 初始化 B 为零，A 使用 Kaiming uniform
            nn.init.zeros_(self.lora_B.weight)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

        # 缩放因子
        self.scaling = alpha / r

    def forward(self, x):
        # 1. 计算原始权重 W_0 的输出 (使用 functional API 和不可训练的权重)
        if self.is_linear:
            # nn.functional.linear(input, weight, bias=None)
            original_output = F.linear(x, self.W_0_weight, self.W_0_bias)

            # LoRA path
            lora_output = (
                              # 矩阵乘法
                                  torch.matmul(x, self.lora_A.transpose(0, 1))
                                  @ self.lora_B.transpose(0, 1)
                          ) * self.scaling

        else:
            # nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
            original_output = F.conv2d(
                x, self.W_0_weight, self.W_0_bias,
                self.stride, self.padding, dilation=1, groups=self.groups
            )

            # LoRA path
            lora_output = self.lora_B(self.lora_A(x)) * self.scaling

        return original_output + lora_output


# ==============================================================================
# --- 配置参数 (新增 LoRA 相关的配置) ---
# ==============================================================================

# --- 配置参数 ---
CONFIG = {
    'MODEL_NAME': 'seresnext50_32x4d',
    'TRANSFER_LEARNING_PLAN': 'PlanC',  #PlanA/PlanB/PlanC
    'ORIGIN_MODEL_ROOT': 'result/origin/model',
    'LOAD_MODEL_WEIGHTS': True,
    'DATA_DIR': './data/data_target',
    'ROOT': 'target',
    'NUM_CLASSES': 5,
    'INPUT_SIZE': 224,
    'BATCH_SIZE': 32,
    'LEARNING_RATE_PLANA': 1e-6,
    'LEARNING_RATE_PLANB': 1e-5,
    'LEARNING_RATE_PLANC': 1e-4, # Plan C (LoRA) 通常使用较高的学习率
    'NUM_EPOCHS': 20,
    'TRAIN_RATIO': 0.6,
    'VAL_RATIO': 0.2,
    'TEST_RATIO': 0.2,
    'RANDOM_SEED': 42,
    'MAX_GRAD_NORM': 1.0,
    'TSNE_SAMPLES': 2000,
    'EARLY_STOPPING_PATIENCE': 5,
    'FREEZE_LAYERS_B': ['conv1', 'bn1', 'layer1', 'layer2'],
    # --- 新增 LoRA 配置 ---
    'LORA_RANK': 4,
    'LORA_ALPHA': 8,
    'LORA_TARGET_MODULES': ['fc', 'conv1', 'conv3', 'conv_reduce', 'conv_expand'],
}

# --- 路径定义 (PlanA/PlanB/PlanC 为父级文件夹) ---
MODEL_SAVE_DIR = f"result/{CONFIG['ROOT']}/{CONFIG['TRANSFER_LEARNING_PLAN']}/model/{CONFIG['MODEL_NAME']}"
REPORT_SAVE_DIR = f"result/{CONFIG['ROOT']}/{CONFIG['TRANSFER_LEARNING_PLAN']}/report/{CONFIG['MODEL_NAME']}"

BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
ORIGIN_WEIGHTS_PATH = os.path.join(CONFIG['ORIGIN_MODEL_ROOT'], CONFIG['MODEL_NAME'], "best_model.pth")

# timm seresnext50_32x4d 对应的层名称
LAYER_NAMES = {
    'initial': 'conv1',
    'middle1': 'layer2',
    'middle2': 'layer3',
    'final': 'global_pool',
}

# --- 全局数据收集变量 ---
HISTORY = {
    'train_loss': [], 'val_loss': [], 'test_loss': [],
    'train_acc': [], 'val_acc': [], 'test_acc': [],
    'train_f1': [], 'val_f1': [], 'test_f1': []
}


# ==============================================================================
# --- 新增 LoRA 模块替换函数 ---
# ==============================================================================

def replace_linear_with_lora(model, r, alpha, target_module_names):
    """
    递归遍历模型，将目标名称的 nn.Linear 或 1x1 nn.Conv2d 替换为 LoRA_Wrapper。
    """
    for name, module in model.named_children():

        target_found = False

        # Case 1: nn.Linear module
        if isinstance(module, nn.Linear) and name in target_module_names:
            print(f"   -> LoRA 替换 Linear层: {name}")
            setattr(model, name, LoRA_Wrapper(module, r=r, alpha=alpha))
            target_found = True

        # Case 2: nn.Conv2d (1x1) module
        elif isinstance(module, nn.Conv2d) and name in target_module_names:
            # 检查 kernel_size 是否为 1x1
            is_1x1 = (module.kernel_size == (1, 1)) or (isinstance(module.kernel_size, int) and module.kernel_size == 1)
            if is_1x1:
                print(f"   -> LoRA 替换 Conv1x1层: {name}")
                setattr(model, name, LoRA_Wrapper(module, r=r, alpha=alpha))
                target_found = True

        # 递归处理子模块
        if not target_found and len(list(module.children())) > 0:
            replace_linear_with_lora(module, r, alpha, target_module_names)


# ==============================================================================
# --- 策略设置 (set_transfer_learning_strategy) ---
# 【新增 PlanC 逻辑，采用修正后的冻结策略】
# ==============================================================================

def set_transfer_learning_strategy(model, plan):
    """
    根据迁移学习策略 (PlanA/PlanB/PlanC) 冻结或解冻模型参数。
    """

    # 1. 重置所有参数的 requires_grad
    for param in model.parameters():
        param.requires_grad = True # 默认开启

    if plan == 'PlanA':
        print("\n--- 策略: Plan A (全微调) ---")
        # Plan A: 全部微调，不冻结任何参数
        return

    elif plan == 'PlanB':
        print(f"\n--- 策略: Plan B (部分冻结微调) ---")

        # 1. 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False

        # 2. 解冻指定层（即 CONFIG['FREEZE_LAYERS_B'] 之外的层）
        freeze_layers = CONFIG['FREEZE_LAYERS_B']
        unfrozen_layers = []
        for name, module in model.named_children():
            # 始终解冻分类器 (如 'fc', 'head' 等) 以及未被指定冻结的层
            if name in ['fc', 'head', 'global_pool'] or name not in freeze_layers:
                for param in module.parameters():
                    param.requires_grad = True
                unfrozen_layers.append(name)

        print(f"   -> 冻结层: {freeze_layers}")
        print(f"   -> 解冻层: {unfrozen_layers}")
        return

    elif plan == 'PlanC':
        print(f"\n--- 策略: Plan C (LoRA 参数高效微调) ---")

        trainable_lora_count = 0

        # 1. 默认冻结所有参数
        for param in model.parameters():
            param.requires_grad = False

        # 2. 解冻 LoRA 矩阵参数
        for name, param in model.named_parameters():
            # 【关键修正】：检查参数名是否包含 'lora_A' 或 'lora_B'
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad = True
                trainable_lora_count += param.numel()

            # (可选) 解冻分类头（如果它没有被 LoRA 替换，且需要微调）
            # 由于在您的配置中 'fc' 也是目标模块，这部分可以省略，确保只有 LoRA 矩阵解冻

        print("   -> 所有原始权重 (W_0) 已被 LoRA_Wrapper 转换为缓冲区/冻结。")
        print(f"   -> 仅 LoRA 矩阵参数 (lora_A/B) 解冻: {trainable_lora_count} 个。")

        # --- 调试检查：打印所有可训练参数 ---
        print("\n--- 详细可训练参数检查 (用于调试) ---")
        total_trainable = 0
        trainable_groups = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                total_trainable += param.numel()
                trainable_groups += 1
                print(f"   -> [可训练] {name} (形状: {list(param.shape)}): {param.numel()} 个参数")
        print(f"总可训练参数: {total_trainable} 个 (参数组数量: {trainable_groups} 个)")

        if total_trainable != trainable_lora_count:
            print(f"!!! 警告: 实际可训练参数 ({total_trainable}) 与统计值 ({trainable_lora_count}) 不符。!!!")

        return

    else:
        raise ValueError(f"未知迁移学习策略: {plan}。请使用 'PlanA', 'PlanB' 或 'PlanC'。")


# ==============================================================================
# --- 模型实例化 (get_model_instance) ---
# 【新增 PlanC 替换逻辑和权重加载兼容性】
# ==============================================================================

def get_model_instance(model_name, num_classes, origin_weights_path, plan):
    """
    初始化模型，并从指定路径加载预训练权重进行迁移学习。
    在 PlanC 模式下，首先替换为 LoRA 模块。
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        model = timm.create_model(model_name.lower(), pretrained=False, num_classes=num_classes)
        print(f"模型 {model_name} 初始化成功 (目标类别数: {num_classes})。")
    except Exception as e:
        print(f"警告：模型 {model_name} 初始化失败。错误: {e}")
        return None

    # 1. 针对 PlanC 策略，替换目标层为 LoRA 层 (发生在加载权重之前)
    if plan == 'PlanC':
        print("\n--- Plan C: 应用 LoRA 模块替换 ---")
        replace_linear_with_lora(
            model,
            r=CONFIG['LORA_RANK'],
            alpha=CONFIG['LORA_ALPHA'],
            target_module_names=CONFIG['LORA_TARGET_MODULES']
        )

    # 2. 加载预训练权重
    if not os.path.exists(origin_weights_path):
        print(f"*** 警告：未找到预训练权重文件: {origin_weights_path}。将使用随机初始化的权重开始训练。***")
        return model

    print(f"\n--- 从 {origin_weights_path} 加载预训练权重 ---")
    try:
        pretrained_dict = torch.load(origin_weights_path, map_location=device)
        model_dict = model.state_dict()

        unmatched_keys_count = 0
        final_load_dict = {}

        for k, v in pretrained_dict.items():

            if plan == 'PlanC':
                # 检查是否是 LoRA 替换的目标层（即 W_0 权重）
                is_lora_target = any(target in k for target in CONFIG['LORA_TARGET_MODULES'])

                if is_lora_target:
                    # 将原始权重/偏置映射到 LoRA_Wrapper 的 W_0_weight/W_0_bias 缓冲区
                    if k.endswith('.weight'):
                        new_k = k.replace('.weight', '.W_0_weight')
                    elif k.endswith('.bias'):
                        new_k = k.replace('.bias', '.W_0_bias')
                    else:
                        new_k = k
                else:
                    new_k = k
            else:
                new_k = k # PlanA/B 保持原样

            if new_k in model_dict and model_dict[new_k].shape == v.shape:
                final_load_dict[new_k] = v
            else:
                unmatched_keys_count += 1

        model.load_state_dict(final_load_dict, strict=False)

        print("   -> 成功加载主体网络权重。")
        if unmatched_keys_count > 0:
            print(f"   -> 分类头或不匹配/跳过权重 ({unmatched_keys_count} 个) 已被随机初始化/跳过。")
        else:
            print("   -> 所有权重均已加载，请确保模型结构和预训练任务匹配。")


    except Exception as e:
        print(f"*** 致命错误：加载预训练权重文件 {origin_weights_path} 失败。错误: {e} ***")
        return None

    return model


# ==============================================================================
# --- 辅助函数：数据处理和统计 (完整实现) ---
# ==============================================================================

def get_data_statistics(dataset: Subset, title: str, class_names: list, verbose=True):
    """计算并打印数据集的类别分布。"""
    if not dataset:
        return Counter()

    # 从 Subset 中获取原始数据集的 targets 和对应的 indices
    targets = np.array(dataset.dataset.targets)[dataset.indices]
    counts = Counter(targets)

    if verbose:
        print(f"\n--- {title} 数据集统计 ({len(dataset)} 个样本) ---")
        for class_idx in sorted(counts.keys()):
            count = counts.get(class_idx, 0)
            if class_idx < len(class_names):
                class_name = class_names[class_idx]
            else:
                class_name = f"未知类别{class_idx}"

            if len(dataset) > 0:
                print(f"  类别 {class_idx} ({class_name}): {count} 个 ({count/len(dataset)*100:.2f}%)")
            else:
                print(f"  类别 {class_idx} ({class_name}): {count} 个 (0.00%)")
        print("---------------------------------------")
    return counts

def calculate_class_weights(train_counts, num_classes):
    """根据训练集类别计数计算类别权重（用于加权交叉熵损失）。"""
    if not train_counts or not max(train_counts.values()):
        return torch.ones(num_classes)

    max_samples = max(train_counts.values())
    weights = torch.zeros(num_classes)

    for i in range(num_classes):
        count = train_counts.get(i, 0)
        if count > 0:
            weights[i] = max_samples / count
        else:
            weights[i] = max_samples # 对于训练集中没有出现的类别，给一个默认高权重

    if weights.sum() > 0:
        weights = weights / weights.sum() * num_classes

    return weights.float()

def extract_frame_id(path):
    """从文件名中提取唯一的帧/批次 ID"""
    filename = os.path.basename(path)
    match = re.match(r'(F\d+)_', filename)
    if match:
        return match.group(1)
    return filename.split('_')[0]

def prepare_data(data_dir, train_ratio, val_ratio, test_ratio, seed):
    """
    加载数据，按'帧ID+类别'进行批次划分，计算类别权重。
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    transform = transforms.Compose([
        transforms.Resize((CONFIG['INPUT_SIZE'], CONFIG['INPUT_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    if len(full_dataset) == 0:
        raise FileNotFoundError(f"未在 {data_dir} 找到任何图像数据。")

    class_names = full_dataset.classes
    CONFIG['NUM_CLASSES'] = len(class_names)

    # 1. 按 Frame ID 和 Class ID 分组
    batch_map = {}
    for i, (path, class_id) in enumerate(tqdm(full_dataset.imgs, desc="分组文件")):
        frame_id = extract_frame_id(path)
        unique_batch_id = f"{class_id}_{frame_id}"
        if unique_batch_id not in batch_map:
            batch_map[unique_batch_id] = []
        batch_map[unique_batch_id].append(i)

    all_batches = list(batch_map.keys())
    if len(all_batches) < 3:
        # 允许程序继续，但发出警告
        print(f"警告：找到的批次数量不足 {len(all_batches)} 个。")

    num_batches = len(all_batches)
    num_train_batches = int(num_batches * train_ratio)
    num_val_batches = int(num_batches * val_ratio)

    # 2. 划分批次
    np.random.seed(seed)
    np.random.shuffle(all_batches)

    train_batches = all_batches[:num_train_batches]
    val_batches = all_batches[num_train_batches:num_train_batches + num_val_batches]
    test_batches = all_batches[num_train_batches + num_val_batches:]

    final_train_indices = [idx for batch_id in train_batches for idx in batch_map[batch_id]]
    final_val_indices = [idx for batch_id in val_batches for idx in batch_map[batch_id]]
    final_test_indices = [idx for batch_id in test_batches for idx in batch_map[batch_id]]

    train_dataset = Subset(full_dataset, final_train_indices)
    val_dataset = Subset(full_dataset, final_val_indices)
    test_dataset = Subset(full_dataset, final_test_indices)

    train_counts = get_data_statistics(train_dataset, "训练集", class_names, verbose=True)
    get_data_statistics(val_dataset, "验证集", class_names, verbose=True)
    get_data_statistics(test_dataset, "测试集", class_names, verbose=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_weights = calculate_class_weights(train_counts, CONFIG['NUM_CLASSES']).to(device)
    print(f"\n计算的类别权重: {class_weights}")


    train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader, class_names, class_weights

def evaluate_and_report(model, data_loader, device, class_names, report_title="Classification Report", get_predictions=False):
    model.eval()
    all_preds = []
    all_labels = []

    if len(data_loader.dataset) == 0:
        return 0.0, all_labels, all_preds if get_predictions else 0.0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0)
    print(f"\n--- {report_title} (Acc: {acc:.4f}, F1-Macro: {f1:.4f}) ---")
    print(report)
    print("----------------------------------------------------------------")

    if get_predictions:
        return f1, all_labels, all_preds
    return f1


def calculate_loss_acc(model, data_loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_corrects = 0

    if len(data_loader.dataset) == 0:
        return 0.0, 0.0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            total_corrects += torch.sum(preds == labels.data)

    loss = total_loss / len(data_loader.dataset)
    acc = total_corrects.double() / len(data_loader.dataset)
    return loss, acc


def train_model(model, train_loader, val_loader, test_loader, device, class_names, class_weights):

    # --- 动态选择学习率 ---
    plan = CONFIG['TRANSFER_LEARNING_PLAN']
    if plan == 'PlanA': lr = CONFIG['LEARNING_RATE_PLANA']
    elif plan == 'PlanB': lr = CONFIG['LEARNING_RATE_PLANB']
    elif plan == 'PlanC': lr = CONFIG['LEARNING_RATE_PLANC']
    else: lr = CONFIG['LEARNING_RATE_PLANB']

    print(f"\n   -> 当前策略 ({plan}) 使用学习率: {lr}")
    # ----------------------

    # 确保只优化 requires_grad=True 的参数
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=lr)

    num_trainable_groups = len(list(filter(lambda p: p.requires_grad, model.parameters())))
    print(f"   -> 优化器将优化 {num_trainable_groups} 个参数组 (未冻结)。")

    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

    best_f1 = 0.0
    max_grad_norm = CONFIG['MAX_GRAD_NORM']

    patience = CONFIG['EARLY_STOPPING_PATIENCE']
    epochs_no_improve = 0
    print(f"\n--- 开始训练 {CONFIG['MODEL_NAME']} ({plan}) (早停耐心值: {patience}) ---")

    for epoch in range(CONFIG['NUM_EPOCHS']):
        model.train()
        running_loss = 0.0
        train_corrects = 0

        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['NUM_EPOCHS']} [TRAIN]", unit="batch")

        for inputs, labels in pbar_train:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            if max_grad_norm is not None:
                # 只对可训练参数进行梯度裁剪
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=max_grad_norm)

            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_corrects += torch.sum(preds == labels.data)
            display_loss = loss.item() if not torch.isnan(loss) else float('nan')
            pbar_train.set_postfix({'Loss': f'{display_loss:.4f}'})

        # 评估阶段
        model.eval()

        print(f"\n*** Epoch {epoch+1} 评估报告 ***")
        train_f1 = evaluate_and_report(model, train_loader, device, class_names, f"训练集 Epoch {epoch+1} Classification Report", get_predictions=False)
        val_f1 = evaluate_and_report(model, val_loader, device, class_names, f"验证集 Epoch {epoch+1} Classification Report", get_predictions=False)
        test_f1 = evaluate_and_report(model, test_loader, device, class_names, f"测试集 Epoch {epoch+1} Classification Report", get_predictions=False)

        if len(train_loader.dataset) == 0:
            epoch_loss = 0.0
            epoch_acc = 0.0
        else:
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = train_corrects.double() / len(train_loader.dataset)

        val_loss, val_acc = calculate_loss_acc(model, val_loader, device, criterion)
        test_loss, test_acc = calculate_loss_acc(model, test_loader, device, criterion)

        # 记录到 HISTORY
        HISTORY['train_loss'].append(epoch_loss)
        HISTORY['train_acc'].append(epoch_acc.item() if isinstance(epoch_acc, torch.Tensor) else epoch_acc)
        HISTORY['train_f1'].append(train_f1)

        HISTORY['val_loss'].append(val_loss)
        HISTORY['val_acc'].append(val_acc.item() if isinstance(val_acc, torch.Tensor) else val_acc)
        HISTORY['val_f1'].append(val_f1)

        HISTORY['test_loss'].append(test_loss)
        HISTORY['test_acc'].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)
        HISTORY['test_f1'].append(test_f1)

        print(f"Epoch {epoch+1}/{CONFIG['NUM_EPOCHS']} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")

        # --- 早停逻辑 ---
        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            # 保存的是包含 LoRA 矩阵和 W_0 缓冲区的完整模型状态
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f" -> 验证集 F1 提升至 {best_f1:.4f}，保存模型到 {BEST_MODEL_PATH}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f" -> 验证集 F1 未提升。当前未提升轮数: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print(f"\n*** 早停触发！连续 {patience} 轮验证集 F1 未提升。停止训练。 ***")
            break
        # ----------------

    return best_f1


def main():
    current_model_save_dir = f"result/{CONFIG['ROOT']}/{CONFIG['TRANSFER_LEARNING_PLAN']}/model/{CONFIG['MODEL_NAME']}"
    current_report_save_dir = f"result/{CONFIG['ROOT']}/{CONFIG['TRANSFER_LEARNING_PLAN']}/report/{CONFIG['MODEL_NAME']}"
    current_best_model_path = os.path.join(current_model_save_dir, "best_model.pth")

    os.makedirs(current_model_save_dir, exist_ok=True)
    os.makedirs(current_report_save_dir, exist_ok=True)

    torch.manual_seed(CONFIG['RANDOM_SEED'])
    np.random.seed(CONFIG['RANDOM_SEED'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"运行设备: {device}")

    try:
        # 1. 准备数据
        train_loader, val_loader, test_loader, class_names, class_weights_tensor = prepare_data(
            CONFIG['DATA_DIR'], CONFIG['TRAIN_RATIO'], CONFIG['VAL_RATIO'], CONFIG['TEST_RATIO'], CONFIG['RANDOM_SEED']
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"数据准备失败: {e}")
        return

    # 2. 初始化模型并加载预训练权重
    model = get_model_instance(
        model_name=CONFIG['MODEL_NAME'],
        num_classes=CONFIG['NUM_CLASSES'],
        origin_weights_path=ORIGIN_WEIGHTS_PATH,
        plan=CONFIG['TRANSFER_LEARNING_PLAN'] # 传递 Plan 以启用 LoRA 替换
    )

    if model is None:
        return

    model = model.to(device)

    # 3. 设置迁移学习策略 (Plan A/B/C)
    if not CONFIG['LOAD_MODEL_WEIGHTS']:
        set_transfer_learning_strategy(model, CONFIG['TRANSFER_LEARNING_PLAN'])

    if CONFIG['LOAD_MODEL_WEIGHTS']:
        print(f"\n--- 跳过训练阶段，直接加载目标任务权重 ({current_best_model_path}) ---")
    else:
        # 4. 训练模型
        train_model(model, train_loader, val_loader, test_loader, device, class_names, class_weights_tensor)

    print(f"\n--- 加载最佳模型进行最终报告 ({current_best_model_path}) ---")

    # 5. 加载最佳模型权重
    if os.path.exists(current_best_model_path):
        try:
            model.load_state_dict(torch.load(current_best_model_path, map_location=device))
            print("   -> 成功加载目标任务的最佳模型权重。")
        except Exception as e:
            print(f"警告: 加载最佳模型文件失败: {e}，使用当前模型状态。")
    else:
        print(f"警告: 找不到目标任务的最佳模型文件，使用当前模型状态。")

    model.eval()

    # 6. 最终报告
    print("\n*** 最终测试集评估报告 ***")
    final_f1, all_labels, all_preds = evaluate_and_report(
        model, test_loader, device, class_names, "最终测试集 Classification Report", get_predictions=True
    )

    # 7. 报告生成
    print("\n--- 报告生成 ---")
    plot_history(HISTORY, CONFIG['MODEL_NAME'], current_report_save_dir, CONFIG['LOAD_MODEL_WEIGHTS'],zoom_metrics=True)
    plot_confusion_matrix(all_labels, all_preds, class_names, CONFIG['MODEL_NAME'], current_report_save_dir)

    try:
        extract_and_plot_tsne(model, test_loader, device, LAYER_NAMES, CONFIG, current_report_save_dir, class_names)
        generate_grad_cam_images(model, test_loader, device, class_names, current_report_save_dir, num_samples=5)
        generate_lime_analysis(model, test_loader, device, class_names, current_report_save_dir, num_samples=5)
    except Exception as e:
        print(f"警告: 报告生成工具运行失败: {e}")


    print("\n所有流程已完成。")


if __name__ == '__main__':
    main()
