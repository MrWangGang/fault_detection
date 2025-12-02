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

try:
    from Reporter import plot_history, plot_confusion_matrix, extract_and_plot_tsne, \
        generate_grad_cam_images, generate_lime_analysis
except ImportError:
    raise ImportError("致命错误：无法导入 Reporter.py 中的报告函数。请确保文件存在且文件名正确。")


# --- 配置参数 ---
CONFIG = {
    'MODEL_NAME': 'resnet50',#seresnext50_32x4d,'resnext50_32x4d','resnet50'
    'LOAD_MODEL_WEIGHTS': True,
    'DATA_DIR': './data/data_origin',
    'ROOT': 'origin',
    'NUM_CLASSES': 5,
    'INPUT_SIZE': 224,
    'BATCH_SIZE': 32,
    'LEARNING_RATE': 1e-5,
    'NUM_EPOCHS': 20,
    'TRAIN_RATIO': 0.6,
    'VAL_RATIO': 0.2,
    'TEST_RATIO': 0.2,
    'RANDOM_SEED': 42,
    'MAX_GRAD_NORM': 1.0,
    'TSNE_SAMPLES': 2000,
    'EARLY_STOPPING_PATIENCE': 5,
}

# --- 路径定义 ---
MODEL_SAVE_DIR = f"result/{CONFIG['ROOT']}/model/{CONFIG['MODEL_NAME']}"
REPORT_SAVE_DIR = f"result/{CONFIG['ROOT']}/report/{CONFIG['MODEL_NAME']}"
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
    'train_f1': [], 'val_f1': [], 'test_f1': []  # <-- 新增 train_f1 和 val_f1
}


def get_model_instance(model_name, pretrained, num_classes):
    try:
        model = timm.create_model(model_name.lower(), pretrained=pretrained, num_classes=num_classes)
        print(f"模型 {model_name} 初始化成功。")
        return model
    except Exception as e:
        print(f"警告：模型 {model_name} 初始化失败。错误: {e}")
        return None


def get_data_statistics(dataset: Subset, title: str, class_names: list, verbose=True):
    if not dataset:
        return Counter()

    targets = np.array(dataset.dataset.targets)[dataset.indices]
    counts = Counter(targets)

    if verbose:
        print(f"\n--- {title} 数据集统计 ({len(dataset)} 个样本) ---")
        for class_idx in sorted(counts.keys()):
            count = counts.get(class_idx, 0)
            class_name = class_names[class_idx]
            print(f"  类别 {class_idx} ({class_name}): {count} 个 ({count/len(dataset)*100:.2f}%)")
        print("---------------------------------------")

    return counts


def calculate_class_weights(train_counts, num_classes):
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


def extract_frame_id(path):
    filename = os.path.basename(path)
    match = re.match(r'(F\d+)_', filename)
    if match:
        return match.group(1)
    return filename.split('_')[0]


def prepare_data(data_dir, train_ratio, val_ratio, test_ratio, seed):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    if len(full_dataset) == 0:
        raise FileNotFoundError(f"未在 {data_dir} 找到任何图像数据。")

    class_names = full_dataset.classes
    CONFIG['NUM_CLASSES'] = len(class_names)

    batch_map = {}
    for i, (path, class_id) in enumerate(tqdm(full_dataset.imgs, desc="分组文件")):
        frame_id = extract_frame_id(path)
        unique_batch_id = f"{class_id}_{frame_id}"
        if unique_batch_id not in batch_map:
            batch_map[unique_batch_id] = []
        batch_map[unique_batch_id].append(i)

    all_batches = list(batch_map.keys())
    if len(all_batches) < 3:
        raise ValueError(f"找到的批次数量不足 {len(all_batches)} 个。")

    num_batches = len(all_batches)
    num_train_batches = int(num_batches * train_ratio)
    num_val_batches = int(num_batches * val_ratio)

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

    train_counts = get_data_statistics(train_dataset, "训练集", class_names, verbose=False)
    get_data_statistics(val_dataset, "验证集", class_names, verbose=False)
    get_data_statistics(test_dataset, "测试集", class_names, verbose=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_weights = calculate_class_weights(train_counts, CONFIG['NUM_CLASSES']).to(device)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=4)

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
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['LEARNING_RATE'])
    best_f1 = 0.0
    max_grad_norm = CONFIG['MAX_GRAD_NORM']
    # --- 早停配置 ---
    patience = CONFIG['EARLY_STOPPING_PATIENCE']
    epochs_no_improve = 0
    print(f"\n--- 开始训练 {CONFIG['MODEL_NAME']} (早停耐心值: {patience}) ---")

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_corrects += torch.sum(preds == labels.data)
            display_loss = loss.item() if not torch.isnan(loss) else float('nan')
            pbar_train.set_postfix({'Loss': f'{display_loss:.4f}'})

        model.eval()

        # 计算 F1 for Train, Validation, Test
        train_f1 = evaluate_and_report(model, train_loader, device, class_names, "训练集 Classification Report")
        val_f1 = evaluate_and_report(model, val_loader, device, class_names, "验证集 Classification Report")
        test_f1 = evaluate_and_report(model, test_loader, device, class_names, "测试集 Classification Report")

        # 计算 Loss/Acc
        if len(train_loader.dataset) == 0:
            epoch_loss = 0.0
            epoch_acc = 0.0
        else:
            # 训练集的 Loss 和 Acc 使用累积值
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = train_corrects.double() / len(train_loader.dataset)

        val_loss, val_acc = calculate_loss_acc(model, val_loader, device, criterion)
        test_loss, test_acc = calculate_loss_acc(model, test_loader, device, criterion)

        # 记录到 HISTORY
        HISTORY['train_loss'].append(epoch_loss)
        HISTORY['train_acc'].append(epoch_acc.item() if isinstance(epoch_acc, torch.Tensor) else epoch_acc)
        HISTORY['train_f1'].append(train_f1) # <-- 记录 Train F1

        HISTORY['val_loss'].append(val_loss)
        HISTORY['val_acc'].append(val_acc.item() if isinstance(val_acc, torch.Tensor) else val_acc)
        HISTORY['val_f1'].append(val_f1) # <-- 记录 Validation F1

        HISTORY['test_loss'].append(test_loss)
        HISTORY['test_acc'].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)
        HISTORY['test_f1'].append(test_f1) # <-- 记录 Test F1

        print(f"Epoch {epoch+1}/{CONFIG['NUM_EPOCHS']} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")

        # --- 早停逻辑 ---
        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f" -> 验证集 F1 提升至 {best_f1:.4f}，保存模型。")
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
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(REPORT_SAVE_DIR, exist_ok=True)

    torch.manual_seed(CONFIG['RANDOM_SEED'])
    np.random.seed(CONFIG['RANDOM_SEED'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        train_loader, val_loader, test_loader, class_names, class_weights_tensor = prepare_data(
            CONFIG['DATA_DIR'], CONFIG['TRAIN_RATIO'], CONFIG['VAL_RATIO'], CONFIG['TEST_RATIO'], CONFIG['RANDOM_SEED']
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"数据准备失败: {e}")
        return

    model = get_model_instance(
        model_name=CONFIG['MODEL_NAME'],
        pretrained=False,
        num_classes=CONFIG['NUM_CLASSES']
    )

    if model is None:
        return

    model = model.to(device)

    if CONFIG['LOAD_MODEL_WEIGHTS']:
        print(f"\n--- 跳过训练阶段，直接加载权重 ---")
    else:
        train_model(model, train_loader, val_loader, test_loader, device, class_names, class_weights_tensor)

    print(f"\n--- 加载最佳模型进行最终报告 ---")

    if os.path.exists(BEST_MODEL_PATH):
        try:
            model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        except Exception as e:
            print(f"警告: 加载最佳模型文件失败: {e}，使用当前模型状态。")
    else:
        print(f"警告: 找不到最佳模型文件，使用当前模型状态。")

    model.eval()

    # 最终报告
    final_f1, all_labels, all_preds = evaluate_and_report(
        model, test_loader, device, class_names, "最终测试集 Classification Report", get_predictions=True
    )

    # 报告生成 (依赖于已修改的 Reporter.py)
    plot_history(HISTORY, CONFIG['MODEL_NAME'], REPORT_SAVE_DIR, CONFIG['LOAD_MODEL_WEIGHTS'])
    plot_confusion_matrix(all_labels, all_preds, class_names, CONFIG['MODEL_NAME'], REPORT_SAVE_DIR)
    extract_and_plot_tsne(model, test_loader, device, LAYER_NAMES, CONFIG, REPORT_SAVE_DIR, class_names)
    generate_grad_cam_images(model, test_loader, device, class_names, REPORT_SAVE_DIR, num_samples=5)
    generate_lime_analysis(model, test_loader, device, class_names, REPORT_SAVE_DIR, num_samples=5)

    print("\n所有流程已完成。")


if __name__ == '__main__':
    main()
