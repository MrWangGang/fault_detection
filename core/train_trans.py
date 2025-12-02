import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report, f1_score, accuracy_score
from collections import Counter
import numpy as np
import os
import re
import timm
from tqdm import tqdm
from itertools import cycle
from torch.autograd import Function


# --- 配置参数 ---
CONFIG = {
    'MODEL_NAME': 'seresnext50_32x4d',
    'TRANSFER_LEARNING_PLAN': 'PlanD',
    'LOAD_MODEL_WEIGHTS': False,

    # --- 数据路径 ---
    'SOURCE_DATA_DIR': './data/data_origin',
    'TARGET_DATA_DIR': './data/data_target',

    'ROOT': 'target',
    'NUM_CLASSES': 5,
    'INPUT_SIZE': 224,
    'BATCH_SIZE': 32,
    'LEARNING_RATE_PlanD': 1e-5,
    'DANN_GAMMA': 1.0,
    'GRL_ALPHA': 1.0,
    'NUM_EPOCHS': 20,
    'TRAIN_RATIO': 0.6,
    'VAL_RATIO': 0.2,
    'TEST_RATIO': 0.2,
    'RANDOM_SEED': 42,
    'MAX_GRAD_NORM': 1.0,
    'EARLY_STOPPING_PATIENCE': 5,
}

# --- 路径定义 ---
MODEL_SAVE_DIR = os.path.join("result", CONFIG['ROOT'], CONFIG['TRANSFER_LEARNING_PLAN'], "model", CONFIG['MODEL_NAME'])
REPORT_SAVE_DIR = os.path.join("result", CONFIG['ROOT'], CONFIG['TRANSFER_LEARNING_PLAN'], "report", CONFIG['MODEL_NAME'])

BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.pth")

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
# --- 辅助函数 ---
# ==============================================================================

def extract_frame_id(path):
    """从文件名中提取帧 ID"""
    filename = os.path.basename(path)
    match = re.match(r'(F\d+)_', filename)
    if match:
        return match.group(1)
    return filename.split('_')[0]


def get_data_statistics(dataset: Subset, title: str, class_names: list, verbose=True):
    """计算数据集中每个类别的样本数量。"""
    if not dataset:
        return Counter()

    # 从 Subset 中获取原始数据集的 targets 和 Subset 的 indices
    targets = np.array(dataset.dataset.targets)[dataset.indices]
    counts = Counter(targets)

    if verbose:
        print(f"\n--- {title} 数据集统计 ({len(dataset)} 个样本) ---")
        for class_idx in sorted(counts.keys()):
            count = counts.get(class_idx, 0)
            class_name = class_names[class_idx] if class_idx < len(class_names) else f"未知类别 {class_idx}"
            print(f"  类别 {class_idx} ({class_name}): {count} 个 ({count/len(dataset)*100:.2f}%)")
        print("---------------------------------------")

    return counts


def calculate_class_weights(train_counts, num_classes):
    """根据训练集样本数计算类别权重"""
    if not train_counts or not max(train_counts.values()):
        return torch.ones(num_classes)

    max_samples = max(train_counts.values())
    weights = torch.zeros(num_classes)
    for i in range(num_classes):
        count = train_counts.get(i, 0)
        if count > 0:
            weights[i] = max_samples / count
        else:
            weights[i] = 1.0

    weights = weights / weights.sum() * num_classes
    return weights.float()


# ==============================================================================
# --- DANN 核心组件：梯度反转层 (GRL) ---
# ==============================================================================

class GradientReversalFunction(Function):
    """
    前向传播时返回输入，反向传播时将梯度乘以 -alpha。
    """
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.alpha = alpha
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    """ GRL 模块 """
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


# ==============================================================================
# --- 模型和策略修改 ---
# ==============================================================================

def set_transfer_learning_strategy(model, plan):
    """
    PlanD: 特征对齐，全部微调
    """
    if plan == 'PlanD':
        print("\n--- 策略: Plan C DANN (对抗性对齐/全微调) ---")
        # 全部微调
        for param in model.parameters():
            param.requires_grad = True
        return
    else:
        raise ValueError(f"未知迁移学习策略: {plan}")


def get_model_instance(model_name, num_classes):
    """
    初始化模型，并包装为 FeatureExtractor 以返回特征、分类输出和领域判别输出。
    """

    print(f"\n--- 初始化模型 {model_name} (随机初始化) ---")

    model = timm.create_model(model_name.lower(), pretrained=False, num_classes=num_classes)
    print(f"模型 {model_name} 初始化成功 (目标类别数: {num_classes})。")

    # DANN Feature Extractor 包装器
    class FeatureExtractor(nn.Module):
        def __init__(self, model):
            super(FeatureExtractor, self).__init__()

            # 特征提取器 F: 主干网络
            self.model_features = nn.Sequential(*list(model.children())[:-2])
            self.global_pool = model.global_pool

            # 分类器 C: 目标任务分类头
            self.task_classifier = model.fc

            # 领域判别器 D: 接收特征输入
            feature_dim = self.task_classifier.in_features
            self.domain_classifier = nn.Sequential(
                GradientReversalLayer(alpha=CONFIG['GRL_ALPHA']),
                nn.Linear(feature_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(512, 2) # 输出 2 个类别
            )

        def forward(self, x):
            x = self.model_features(x)
            features = self.global_pool(x)

            # 确保特征扁平化
            if features.dim() > 2:
                features = features.view(features.size(0), -1)

                # 分类任务输出
            task_output = self.task_classifier(features)

            # 领域判别输出
            domain_output = self.domain_classifier(features)

            return task_output, features, domain_output

    feature_model = FeatureExtractor(model)
    return feature_model


# ==============================================================================
# --- 数据准备 ---
# ==============================================================================

def prepare_data_PlanD(source_data_dir, target_data_dir, train_ratio, val_ratio, test_ratio, seed):
    """
    准备源域和目标域数据。
    """
    transform = transforms.Compose([
        transforms.Resize((CONFIG['INPUT_SIZE'], CONFIG['INPUT_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 1. 加载目标域数据并划分
    target_full_dataset = datasets.ImageFolder(target_data_dir, transform=transform)
    if len(target_full_dataset) == 0:
        raise RuntimeError(f"未在 {target_data_dir} 找到任何目标域图像数据。")

    class_names = target_full_dataset.classes
    CONFIG['NUM_CLASSES'] = len(class_names)

    # 目标域数据划分逻辑（基于帧 ID 的分组划分）
    batch_map = {}
    for i, (path, class_id) in enumerate(tqdm(target_full_dataset.imgs, desc="分组目标域文件")):
        frame_id = extract_frame_id(path)
        unique_batch_id = f"{class_id}_{frame_id}"
        if unique_batch_id not in batch_map:
            batch_map[unique_batch_id] = []
        batch_map[unique_batch_id].append(i)

    all_batches = list(batch_map.keys())
    if len(all_batches) < 3:
        # 如果批次不足，则改为普通随机划分
        print("警告: 目标域批次数量不足 3 个，改为随机划分。")
        num_samples = len(target_full_dataset)
        num_train = int(num_samples * train_ratio)
        num_val = int(num_samples * val_ratio)
        num_test = num_samples - num_train - num_val
        indices = torch.randperm(num_samples).tolist()

        target_train_indices = indices[:num_train]
        target_val_indices = indices[num_train:num_train + num_val]
        target_test_indices = indices[num_train + num_val:]
    else:
        num_batches = len(all_batches)
        num_train_batches = int(num_batches * train_ratio)
        num_val_batches = int(num_batches * val_ratio)

        np.random.seed(seed)
        np.random.shuffle(all_batches)

        train_batches = all_batches[:num_train_batches]
        val_batches = all_batches[num_train_batches:num_train_batches + num_val_batches]
        test_batches = all_batches[num_train_batches + num_val_batches:]

        target_train_indices = [idx for batch_id in train_batches for idx in batch_map[batch_id]]
        target_val_indices = [idx for batch_id in val_batches for idx in batch_map[batch_id]]
        target_test_indices = [idx for batch_id in test_batches for idx in batch_map[batch_id]]

    target_train_dataset = Subset(target_full_dataset, target_train_indices)
    target_val_dataset = Subset(target_full_dataset, target_val_indices)
    target_test_dataset = Subset(target_full_dataset, target_test_indices)

    target_train_counts = get_data_statistics(target_train_dataset, "目标域训练集", class_names, verbose=True)
    get_data_statistics(target_val_dataset, "目标域验证集", class_names, verbose=True)
    get_data_statistics(target_test_dataset, "目标域测试集", class_names, verbose=True)

    # 2. 加载源域数据
    source_dataset = datasets.ImageFolder(source_data_dir, transform=transform)
    if len(source_dataset) == 0:
        print(f"警告：未在 {source_data_dir} 找到任何源域图像数据。DANN 将退化为普通分类器。")
        source_train_loader = None
    else:
        source_train_dataset = Subset(source_dataset, list(range(len(source_dataset))))
        get_data_statistics(source_train_dataset, "源域训练集", class_names, verbose=True)
        source_train_loader = DataLoader(source_train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=4, pin_memory=True)

    # 3. 计算类别权重
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_weights = calculate_class_weights(target_train_counts, CONFIG['NUM_CLASSES']).to(device)
    print(f"\n计算的类别权重 (基于目标域训练集): {class_weights}")

    # 4. 创建 DataLoaders
    target_train_loader = DataLoader(target_train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=4, pin_memory=True)
    target_val_loader = DataLoader(target_val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=4, pin_memory=True)
    target_test_loader = DataLoader(target_test_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=4, pin_memory=True)

    return source_train_loader, target_train_loader, target_val_loader, target_test_loader, class_names, class_weights


# ==============================================================================
# --- 评估/损失计算 ---
# ==============================================================================

def calculate_loss_acc(model, data_loader, device, criterion):
    # 适应 model 的输出 (task_output, features, domain_output)
    model.eval()
    total_loss = 0.0
    total_corrects = 0

    if len(data_loader.dataset) == 0:
        return 0.0, 0.0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs, _, _ = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            total_corrects += torch.sum(preds == labels.data)

    loss = total_loss / len(data_loader.dataset)
    acc = total_corrects.double() / len(data_loader.dataset)
    return loss, acc


def evaluate_and_report(model, data_loader, device, class_names, report_title="Classification Report", get_predictions=False):
    # 适应 model 的输出 (task_output, features, domain_output)
    model.eval()
    all_preds = []
    all_labels = []

    if len(data_loader.dataset) == 0:
        return 0.0, all_labels, all_preds if get_predictions else 0.0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs, _, _ = model(inputs)
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


# ==============================================================================
# --- 训练函数 (DANN 核心逻辑) ---
# ==============================================================================

def train_model_PlanD(model, source_loader, target_loader, val_loader, test_loader, device, class_names, class_weights):

    lr = CONFIG['LEARNING_RATE_PlanD']
    dann_gamma = CONFIG['DANN_GAMMA']
    print(f"\n   -> 当前策略 (PlanD/DANN) 使用学习率: {lr}, DANN 损失权重: {dann_gamma}")

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=lr)
    print(f"   -> 优化器将优化 {len(list(filter(lambda p: p.requires_grad, model.parameters())))} 个参数组。")

    # 1. 任务分类损失
    classification_criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    # 2. 领域判别损失
    domain_criterion = nn.CrossEntropyLoss().to(device)

    if source_loader is None:
        source_iterator = None
    else:
        source_iterator = cycle(source_loader)

    best_f1 = 0.0
    max_grad_norm = CONFIG['MAX_GRAD_NORM']
    patience = CONFIG['EARLY_STOPPING_PATIENCE']
    epochs_no_improve = 0

    print(f"\n--- 开始训练 {CONFIG['MODEL_NAME']} (PlanD/DANN) (早停耐心值: {patience}) ---")

    for epoch in range(CONFIG['NUM_EPOCHS']):
        model.train()
        running_loss = 0.0
        train_corrects = 0

        pbar_train = tqdm(target_loader, desc=f"Epoch {epoch+1}/{CONFIG['NUM_EPOCHS']} [TRAIN]", unit="batch")

        # 同时处理源域和目标域
        for step, (target_inputs, target_labels) in enumerate(pbar_train):

            optimizer.zero_grad()

            domain_loss = torch.tensor(0.0).to(device)
            cls_loss = torch.tensor(0.0).to(device)
            total_loss = torch.tensor(0.0).to(device)

            if source_iterator is not None:

                try:
                    source_inputs, _ = next(source_iterator)
                except StopIteration:
                    source_iterator = cycle(source_loader)
                    source_inputs, _ = next(source_iterator)

                source_inputs = source_inputs.to(device)
                target_inputs_device = target_inputs.to(device)
                target_labels_device = target_labels.to(device)

                # 源域标签: 0
                source_domain_labels = torch.zeros(source_inputs.size(0), dtype=torch.long).to(device)
                # 目标域标签: 1
                target_domain_labels = torch.ones(target_inputs_device.size(0), dtype=torch.long).to(device)

                # 拼接输入和领域标签
                all_inputs = torch.cat([source_inputs, target_inputs_device], dim=0)
                all_domain_labels = torch.cat([source_domain_labels, target_domain_labels], dim=0)

                # DANN 前向传播
                task_output_all, _, domain_output_all = model(all_inputs)

                # a) 领域判别损失
                domain_loss = domain_criterion(domain_output_all, all_domain_labels)

                # b) 任务分类损失 (只针对目标域数据)
                target_task_output = task_output_all[source_inputs.size(0):]
                cls_loss = classification_criterion(target_task_output, target_labels_device)

                # c) 总损失
                total_loss = cls_loss + dann_gamma * domain_loss

                # 用于记录训练准确率的预测
                _, preds = torch.max(target_task_output.data, 1)

            else:
                # 退化为普通分类器
                target_inputs_device = target_inputs.to(device)
                target_labels_device = target_labels.to(device)
                cls_output, _, _ = model(target_inputs_device)
                cls_loss = classification_criterion(cls_output, target_labels_device)
                total_loss = cls_loss

                # 用于记录训练准确率的预测
                _, preds = torch.max(cls_output.data, 1)

            # 反向传播和优化
            total_loss.backward()

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            optimizer.step()

            # 记录指标 (仅基于目标域数据)
            running_loss += total_loss.item() * target_inputs.size(0)
            train_corrects += torch.sum(preds == target_labels_device.data)


            display_loss = total_loss.item() if not torch.isnan(total_loss) else float('nan')
            pbar_train.set_postfix({'TotalLoss': f'{display_loss:.4f}', 'CLS': f'{cls_loss.item():.4f}', 'DANN': f'{domain_loss.item():.4f}'})


        # 评估阶段
        model.eval()

        print(f"\n*** Epoch {epoch+1} 评估报告 ***")
        train_f1 = evaluate_and_report(model, target_loader, device, class_names, f"训练集(目标域) Epoch {epoch+1} Classification Report", get_predictions=False)
        val_f1 = evaluate_and_report(model, val_loader, device, class_names, f"验证集 Epoch {epoch+1} Classification Report", get_predictions=False)
        test_f1 = evaluate_and_report(model, test_loader, device, class_names, f"测试集 Epoch {epoch+1} Classification Report", get_predictions=False)

        if len(target_loader.dataset) == 0:
            epoch_loss = 0.0
            epoch_acc = 0.0
        else:
            epoch_loss = running_loss / len(target_loader.dataset)
            epoch_acc = train_corrects.double() / len(target_loader.dataset)

        val_loss, val_acc = calculate_loss_acc(model, val_loader, device, classification_criterion)
        test_loss, test_acc = calculate_loss_acc(model, test_loader, device, classification_criterion)

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

        print(f"Epoch {epoch+1}/{CONFIG['NUM_EPOCHS']} | Train Total Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")

        # --- 早停逻辑 (基于验证集 F1) ---
        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
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
    # 确保路径中的所有父目录都存在
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(REPORT_SAVE_DIR, exist_ok=True)

    torch.manual_seed(CONFIG['RANDOM_SEED'])
    np.random.seed(CONFIG['RANDOM_SEED'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"运行设备: {device}")

    try:
        # 1. 准备数据
        source_train_loader, target_train_loader, val_loader, test_loader, class_names, class_weights_tensor = prepare_data_PlanD(
            CONFIG['SOURCE_DATA_DIR'], CONFIG['TARGET_DATA_DIR'],
            CONFIG['TRAIN_RATIO'], CONFIG['VAL_RATIO'], CONFIG['TEST_RATIO'], CONFIG['RANDOM_SEED']
        )
    except RuntimeError as e:
        print(f"数据准备失败: {e}")
        return

    # 2. 初始化模型
    model = get_model_instance(
        model_name=CONFIG['MODEL_NAME'],
        num_classes=CONFIG['NUM_CLASSES']
    )

    if model is None:
        return

    model = model.to(device)

    # 3. 设置迁移学习策略
    set_transfer_learning_strategy(model, CONFIG['TRANSFER_LEARNING_PLAN'])

    if CONFIG['LOAD_MODEL_WEIGHTS']:
        print(f"\n--- 跳过训练阶段，直接加载目标任务权重 ({BEST_MODEL_PATH}) ---")
    else:
        # 4. 训练模型
        train_model_PlanD(model, source_train_loader, target_train_loader, val_loader, test_loader, device, class_names, class_weights_tensor)

    print(f"\n--- 加载最佳模型进行最终报告 ({BEST_MODEL_PATH}) ---")

    # 5. 加载最佳模型权重
    if os.path.exists(BEST_MODEL_PATH):
        try:
            model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
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

    print("\n所有流程已完成。")


if __name__ == '__main__':
    main()
