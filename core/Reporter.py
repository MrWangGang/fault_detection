import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import json

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from skimage.segmentation import mark_boundaries
    from lime import lime_image
    LIME_GRADCAM_AVAILABLE = True
except ImportError:
    LIME_GRADCAM_AVAILABLE = False


extracted_features_global = {}

def hook_fn_collector(module, input, output, name):
    global extracted_features_global
    if output.ndim > 2:
        features = output.view(output.size(0), -1)
    else:
        features = output
    extracted_features_global[name].append(features.cpu().numpy())


def denormalize_img(img_np):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    denorm_img = img_np * std + mean
    return np.clip(denorm_img, 0, 1)


def get_original_filepath(dataset_subset, index_in_subset):
    """
    根据 Subset 中的索引获取原始数据集中的文件路径。
    """
    if index_in_subset >= len(dataset_subset.indices):
        global_index = index_in_subset
    else:
        global_index = dataset_subset.indices[index_in_subset]

    path, _ = dataset_subset.dataset.imgs[global_index]
    return os.path.basename(path)


def save_history_to_json(history, report_save_dir):
    """保存训练历史记录到 JSON 文件。"""
    history_path = os.path.join(report_save_dir, "history_metrics.json")

    # 确保 JSON 数据是可序列化的
    history_serializable = {k: [float(x) for x in v] for k, v in history.items()}

    try:
        os.makedirs(report_save_dir, exist_ok=True)
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=4)
        print(f"已保存训练历史记录到 JSON 文件: {history_path}")
    except Exception as e:
        print(f"警告: 无法保存历史记录到 JSON 文件: {e}")

def load_history_from_json(report_save_dir):
    """从 JSON 文件加载训练历史记录。"""
    history_path = os.path.join(report_save_dir, "history_metrics.json")
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            print(f"已从 JSON 文件成功加载历史记录。")
            return history
        except Exception as e:
            print(f"警告: 无法从 JSON 文件加载历史记录: {e}")
            return None
    return None

def plot_history(history, model_name, report_save_dir, load_weights_flag, zoom_metrics=False):
    """
    绘制训练历史曲线。
    新增参数 zoom_metrics=True 时，ACC/F1 Y轴范围为 [0.9, 1.1]。
    zoom_metrics=False 时，ACC/F1 Y轴范围为 [0, 1.2]。
    """
    data_to_plot = history

    if load_weights_flag:
        # 1. 加载权重模式，尝试从 JSON 加载数据
        loaded_history = load_history_from_json(report_save_dir)
        if loaded_history and loaded_history.get('train_loss'):
            data_to_plot = loaded_history
        else:
            print(f"警告: 加载权重模式下，未找到或 JSON 数据为空，跳过曲线绘制。")
            return
    elif not history.get('train_loss'):
        # 2. 训练模式但没有数据，跳过
        return
    else:
        # 3. 训练模式且有数据，先保存 JSON
        save_history_to_json(data_to_plot, report_save_dir)


    epochs = range(1, len(data_to_plot['train_loss']) + 1)
    plt.figure(figsize=(18, 5))

    # 最大的 Epoch 数用于 X 轴自适应的上限（略微多一点）
    max_epoch = len(data_to_plot['train_loss']) + 1

    # ----------------------------------------------------
    # --- 【修改】定义 Y 轴刻度 ---

    # 1. 针对 Accuracy 和 F1-Score (全范围模式: 0.0 - 1.2)
    y_ticks_acc_f1_full = np.array([
        0.0, 0.2,0.4,0.6,0.8,1.0,1.2
    ])
    y_labels_acc_f1_full = [f"{t:.2f}" for t in y_ticks_acc_f1_full]

    # 2. 针对 Accuracy 和 F1-Score (缩放模式: 0.9 - 1.1)
    y_ticks_acc_f1_zoom = np.array([
        0.90, 0.92, 0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10
    ])
    y_labels_acc_f1_zoom = [f"{t:.2f}" for t in y_ticks_acc_f1_zoom]

    # 3. 针对 Loss (固定模式)
    y_ticks_loss = np.array([
        0.0, 0.2, 0.4, 0.6, 0.8, 1.0,
        1.5, 2.0, 2.5, 3.0
    ])
    y_labels_loss = [f"{t:.1f}" for t in y_ticks_loss]
    # ----------------------------------------------------


    # Loss 曲线 (Train, Validation, Test)
    plt.subplot(1, 3, 1)
    plt.plot(epochs, data_to_plot['train_loss'], label='Train Loss', markersize=5)
    plt.plot(epochs, data_to_plot['val_loss'], label='Validation Loss', markersize=4)
    if 'test_loss' in data_to_plot and data_to_plot['test_loss']:
        plt.plot(epochs, data_to_plot['test_loss'], label='Test Loss', markersize=4)
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # --- 设定 Loss 曲线的 Y 轴刻度和范围 ---
    plt.ylim(0, 3.2) # Y 轴: 0 - 3.2
    plt.yticks(y_ticks_loss, y_labels_loss)
    plt.xlim(0, max_epoch) # X 轴: 自适应
    # ------------------------------------

    # Accuracy 曲线 (Train, Validation, Test)
    plt.subplot(1, 3, 2)
    plt.plot(epochs, data_to_plot['train_acc'], label='Train Accuracy', markersize=5)
    plt.plot(epochs, data_to_plot['val_acc'], label='Validation Accuracy', markersize=4)
    if 'test_acc' in data_to_plot and data_to_plot['test_acc']:
        plt.plot(epochs, data_to_plot['test_acc'], label='Test Accuracy', markersize=4)
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # --- 设定 Accuracy 曲线的 Y 轴刻度和范围 (根据 zoom_metrics) ---
    if zoom_metrics:
        plt.ylim(0.9, 1.1) # Y 轴: 0.9 - 1.1 (缩放模式)
        plt.yticks(y_ticks_acc_f1_zoom, y_labels_acc_f1_zoom)
    else:
        plt.ylim(0, 1.2) # Y 轴: 0 - 1.2 (全范围模式)
        plt.yticks(y_ticks_acc_f1_full, y_labels_acc_f1_full)
    plt.xlim(0, max_epoch) # X 轴: 自适应
    # ------------------------------------

    # F1-Macro 曲线 (Train, Validation, Test)
    plt.subplot(1, 3, 3)

    # 绘制 Train F1
    if 'train_f1' in data_to_plot and data_to_plot['train_f1']:
        plt.plot(epochs, data_to_plot['train_f1'], label='Train F1-Macro Score', markersize=5, color='blue')

    # 绘制 Validation F1
    if 'val_f1' in data_to_plot and data_to_plot['val_f1']:
        plt.plot(epochs, data_to_plot['val_f1'], label='Validation F1-Macro Score', markersize=4, color='orange')

    # 绘制 Test F1
    if 'test_f1' in data_to_plot and data_to_plot['test_f1']:
        plt.plot(epochs, data_to_plot['test_f1'], label='Test F1-Macro Score', markersize=4, color='green')

    plt.title('F1-Score Curve')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.legend()

    # --- 设定 F1-Score 曲线的 Y 轴刻度和范围 (根据 zoom_metrics) ---
    if zoom_metrics:
        plt.ylim(0.9, 1.1) # Y 轴: 0.9 - 1.1 (缩放模式)
        plt.yticks(y_ticks_acc_f1_zoom, y_labels_acc_f1_zoom)
    else:
        plt.ylim(0, 1.2) # Y 轴: 0 - 1.2 (全范围模式)
        plt.yticks(y_ticks_acc_f1_full, y_labels_acc_f1_full)
    plt.xlim(0, max_epoch) # X 轴: 自适应
    # ------------------------------------

    plt.tight_layout()
    os.makedirs(report_save_dir, exist_ok=True)
    plt.savefig(os.path.join(report_save_dir, f"{model_name}_metrics_curves.png"))
    plt.close()
    print(f"已保存 Loss/Acc/F1 曲线图到 {report_save_dir}/")


def plot_confusion_matrix(all_labels, all_preds, class_names, model_name, report_save_dir):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name} (Test Set)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    os.makedirs(report_save_dir, exist_ok=True)
    plt.savefig(os.path.join(report_save_dir, f"{model_name}_confusion_matrix.png"))
    plt.close()
    print(f"已保存混淆矩阵到 {report_save_dir}/")


def extract_and_plot_tsne(model, test_loader, device, layer_names, config, report_save_dir, class_names):
    global extracted_features_global
    model.eval()

    hooks = []
    modules = dict(model.named_modules())
    extracted_features_global = {k: [] for k in layer_names.keys()}

    for key, layer_name in layer_names.items():
        module = modules.get(layer_name)
        if module is not None:
            hook = module.register_forward_hook(
                lambda module, input, output, k=key: hook_fn_collector(module, input, output, k)
            )
            hooks.append(hook)

    all_labels = []
    total_samples = len(test_loader.dataset)
    tsne_samples = min(config['TSNE_SAMPLES'], total_samples)

    if tsne_samples == 0:
        return

    np.random.seed(config['RANDOM_SEED'])
    all_indices = test_loader.dataset.indices
    sample_indices = np.random.choice(all_indices, tsne_samples, replace=False)

    full_dataset = test_loader.dataset.dataset
    sampler_dataset = Subset(full_dataset, sample_indices)
    sample_loader = DataLoader(sampler_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=4)

    pbar = tqdm(sample_loader, desc=f"T-SNE 提取特征 ({tsne_samples} 样本)", unit="batch")

    with torch.no_grad():
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            _ = model(inputs)
            all_labels.extend(labels.cpu().numpy())

    for hook in hooks:
        hook.remove()

    final_features = {}
    for key in layer_names.keys():
        if extracted_features_global[key]:
            features_combined = np.concatenate(extracted_features_global[key], axis=0)
            final_features[key] = features_combined[:tsne_samples]

    all_labels = np.array(all_labels)[:tsne_samples]

    features_to_plot = {k: v for k, v in final_features.items() if v.size > 0}
    if not features_to_plot:
        return

    plt.figure(figsize=(15, 12))
    layer_titles = {
        'initial': 'Initial Layer (Conv1) [PCA Reduced]',
        'middle1': 'Middle Layer 1 (Layer2)',
        'middle2': 'Middle Layer 2 (Layer3)',
        'final': 'Classification Layer (Global Pool Feature)'
    }
    plot_index = 1

    for key, features in features_to_plot.items():
        print(f"--- 正在对 {layer_titles.get(key, key)} 进行 T-SNE 降维...")

        if features.shape[1] > 2048:
            if features.shape[0] > 100:
                n_components = min(features.shape[0] - 1, 100)
                pca = PCA(n_components=n_components, random_state=config['RANDOM_SEED'])
                features = pca.fit_transform(features)
                print(f"    -> 已使用 PCA 降维至 {features.shape[1]} 维。")
            else:
                print("    -> 样本数过少，跳过 PCA 降维。")
                continue

        try:
            tsne = TSNE(n_components=2, random_state=config['RANDOM_SEED'], perplexity=30.0, max_iter=1000, learning_rate='auto')

            if features.shape[0] < tsne.n_components + 1:
                continue
            features_2d = tsne.fit_transform(features)
        except Exception as e:
            print(f"错误: T-SNE 降维失败 ({key}): {e}")
            continue

        plt.subplot(2, 2, plot_index)
        sns.scatterplot(
            x=features_2d[:, 0], y=features_2d[:, 1], hue=all_labels,
            palette=sns.color_palette("hsv", len(class_names)), legend="full", alpha=0.6
        )
        plt.title(f'T-SNE Visualization: {layer_titles.get(key, key)}')
        plt.legend(class_names, title="Classes", loc='best', fontsize='small')
        plt.grid(True, alpha=0.3)
        plot_index += 1

    if plot_index > 1:
        plt.tight_layout()
        tsne_save_path = os.path.join(report_save_dir, f"{config['MODEL_NAME']}_tsne_visualization.png")
        plt.savefig(tsne_save_path)
        plt.close()
        print(f"已保存 T-SNE 可视化图到 {report_save_dir}/")


def select_analysis_samples(model, data_loader, device, num_classes, confidence_threshold=0.8):
    """筛选高置信度正确和高置信度错误样本的索引。"""
    model.eval()
    all_results = []

    global_indices = data_loader.dataset.indices

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probas = torch.softmax(outputs, dim=1)
            max_probas, preds = torch.max(probas, 1)

            start_idx = i * data_loader.batch_size
            current_batch_global_indices = global_indices[start_idx: start_idx + len(inputs)]

            for j in range(len(inputs)):
                all_results.append({
                    'global_index': current_batch_global_indices[j],
                    'true_label': labels[j].item(),
                    'pred_label': preds[j].item(),
                    'max_proba': max_probas[j].item()
                })

    chc_indices = {}
    ihc_indices = {}

    random.seed(42)
    random.shuffle(all_results)

    for res in all_results:
        g_idx = res['global_index']
        true_label = res['true_label']
        pred_label = res['pred_label']
        max_proba = res['max_proba']

        is_correct = (true_label == pred_label)
        is_high_conf = (max_proba >= confidence_threshold)

        # 1. 筛选高置信度正确样本 (CHC): 每个真实类别一个
        if is_correct and is_high_conf and true_label not in chc_indices:
            chc_indices[true_label] = g_idx

        # 2. 筛选高置信度错误样本 (IHC): 每个预测的错误类别一个
        if not is_correct and is_high_conf and pred_label not in ihc_indices:
            ihc_indices[pred_label] = g_idx

    # 将找到的样本索引和其类型标记合并
    final_indices_map = {}
    for t_idx, g_idx in chc_indices.items():
        final_indices_map[g_idx] = 'correct'
    for p_idx, g_idx in ihc_indices.items():
        final_indices_map[g_idx] = 'wrong'

    final_indices = list(final_indices_map.keys())

    tqdm.write(f"\n[样本筛选] 找到高置信度正确样本 (CHC): {len(chc_indices)} 个")
    tqdm.write(f"[样本筛选] 找到高置信度错误样本 (IHC): {len(ihc_indices)} 个")
    tqdm.write(f"[样本筛选] 最终分析样本数: {len(final_indices)} 个")

    return final_indices, final_indices_map


def generate_grad_cam_images(model, test_loader, device, class_names, report_save_dir, num_samples=5):
    if not LIME_GRADCAM_AVAILABLE:
        print("警告: 缺少 Grad-CAM 库。跳过 Grad-CAM 分析。")
        return

    print("\n--- 正在生成 Grad-CAM 热力图分析 (筛选样本) ---")

    # ------------------------------------------------------------------
    # --- STFT 频谱图轴刻度转换参数（基于用户输入） ---
    IMAGE_SIZE = 224      # 图像的像素尺寸（高和宽的假设值）
    SAMPLE_RATE = 12000   # 信号采样率: 12 kHz
    HOP_LENGTH = 256      # 步长/跳数: 256

    # 重新计算总时长
    TOTAL_TIME_SEC = IMAGE_SIZE * HOP_LENGTH / SAMPLE_RATE # 224 * 256 / 12000 ≈ 4.773s

    TICK_COUNT = 5        # 轴上显示的刻度数量
    # ------------------------------------------------------------------

    # 筛选分析样本
    analysis_indices, index_type_map = select_analysis_samples(model, test_loader, device, len(class_names))

    if not analysis_indices:
        print("警告: 未找到任何满足高置信度条件的分析样本。")
        return

    # 设置保存路径
    cam_dir = os.path.join(report_save_dir, 'grad_cam_combined')
    os.makedirs(os.path.join(cam_dir, 'correct'), exist_ok=True)
    os.makedirs(os.path.join(cam_dir, 'wrong'), exist_ok=True)

    # 重新创建 DataLoader
    full_dataset = test_loader.dataset.dataset
    sample_dataset = Subset(full_dataset, analysis_indices)
    sample_loader = DataLoader(sample_dataset, batch_size=1, shuffle=False)

    target_layers = [model.layer4, model.layer2]
    target_layer_names = ['layer4', 'layer2']
    model.eval()

    # 实例化 GradCAM 算法
    cam_algorithms = {name: GradCAM(model=model, target_layers=[target_layer])
                      for name, target_layer in zip(target_layer_names, target_layers)}

    pbar = tqdm(sample_loader, desc=f"生成 Grad-CAM 组合图 ({len(analysis_indices)} 样本)", unit="sample")

    # --- 轴刻度预计算 ---
    # 像素点位置 (0 到 IMAGE_SIZE-1)
    pixel_ticks = np.linspace(0, IMAGE_SIZE - 1, num=TICK_COUNT, dtype=int)

    # X 轴：时间 (秒)
    time_ticks_sec = pixel_ticks * (TOTAL_TIME_SEC / IMAGE_SIZE)
    time_labels = [f"{t:.2f}s" for t in time_ticks_sec]

    # Y 轴：频率 (kHz)
    max_freq = SAMPLE_RATE / 2 # 6000 Hz
    # 假设 Y=0 对应 Max_Freq，Y=223 对应 0 Hz
    freq_ticks_hz = max_freq * (1 - pixel_ticks / IMAGE_SIZE)
    freq_labels = [f"{f/1000:.1f}kHz" for f in freq_ticks_hz]
    # ---------------------

    for i, (input_tensor, true_label) in enumerate(pbar):

        global_index = analysis_indices[i]
        original_filename = get_original_filepath(test_loader.dataset, global_index)

        # 确保输入张量在正确的设备上
        input_device = input_tensor.to(device)
        true_label = true_label.item()

        with torch.no_grad():
            output = model(input_device) # 使用 input_device
        predicted_class = output.argmax(dim=1).item()

        class_name = class_names[true_label]
        pred_name = class_names[predicted_class]

        # 确定子文件夹
        folder = index_type_map.get(global_index, 'unknown')
        if folder == 'unknown': continue

        file_identifier = os.path.splitext(original_filename)[0]

        # 准备原图（去标准化）
        rgb_img = input_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        rgb_img_denorm = denormalize_img(rgb_img)

        # 收集 CAM 结果
        cam_results = {}
        targets = [ClassifierOutputTarget(predicted_class)]

        for name, cam_algorithm in cam_algorithms.items():
            # 使用已经在设备上的张量 input_device
            grayscale_cam = cam_algorithm(input_tensor=input_device, targets=targets, aug_smooth=True)
            grayscale_cam = grayscale_cam[0, :]
            cam_image = show_cam_on_image(rgb_img_denorm, grayscale_cam, use_rgb=True)
            cam_results[name] = cam_image

        # --- 生成 1x3 组合图 (原图, Layer4 CAM, Layer2 CAM) ---
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), dpi=100) # 调整尺寸

        for idx, ax in enumerate(axes):

            # 1. 绘图内容
            if idx == 0:
                ax.imshow(rgb_img_denorm)
                ax.set_title("Original Spectrogram", fontsize=8)
            elif idx == 1:
                ax.imshow(cam_results['layer4'])
                ax.set_title(f"Grad-CAM (Lyr4)", fontsize=8)
            elif idx == 2:
                ax.imshow(cam_results['layer2'])
                ax.set_title(f"Grad-CAM (Lyr2)", fontsize=8)

            # 2. 应用时间和频率刻度
            ax.set_xticks(pixel_ticks)
            ax.set_xticklabels(time_labels, fontsize=6)
            ax.set_yticks(pixel_ticks)
            ax.set_yticklabels(freq_labels, fontsize=6)
            ax.set_xlabel("Time (s)", fontsize=7)
            ax.set_ylabel("Frequency (kHz)", fontsize=7)
            # 3. 网格
            ax.grid(True, alpha=0.3)


            # 设定总标题并保存
        full_title = f"True: {class_name} | Pred: {pred_name} | File: {file_identifier}"
        fig.suptitle(full_title, fontsize=9)

        # 确定保存路径
        img_filename = f"{file_identifier}_CAM_True_{class_name}_Pred_{pred_name}_Combined.png"
        img_path = os.path.join(cam_dir, folder, img_filename)

        plt.tight_layout(rect=[0, 0, 1, 0.95]) # 为 suptitle 留出空间
        plt.savefig(img_path)
        plt.close(fig)

    print(f"已保存 Grad-CAM 组合图到 {cam_dir}/")


def generate_lime_analysis(model, test_loader, device, class_names, report_save_dir, num_samples=5):
    if not LIME_GRADCAM_AVAILABLE:
        print("警告: 缺少 LIME 库。跳过 LIME 分析。")
        return

    print("\n--- 正在生成 LIME 组合分析图 (筛选样本) ---")

    # ------------------------------------------------------------------
    # --- STFT 频谱图轴刻度转换参数（基于用户输入） ---
    IMAGE_SIZE = 224      # 图像的像素尺寸（高和宽的假设值）
    SAMPLE_RATE = 12000   # 信号采样率: 12 kHz
    HOP_LENGTH = 256      # 步长/跳数: 256

    # 重新计算总时长
    TOTAL_TIME_SEC = IMAGE_SIZE * HOP_LENGTH / SAMPLE_RATE # 224 * 256 / 12000 ≈ 4.773s

    TICK_COUNT = 5        # 轴上显示的刻度数量
    # ------------------------------------------------------------------

    # 筛选分析样本
    analysis_indices, index_type_map = select_analysis_samples(model, test_loader, device, len(class_names))

    if not analysis_indices:
        print("警告: 未找到任何满足高置信度条件的分析样本。")
        return

    # 设置保存路径
    lime_dir = os.path.join(report_save_dir, 'lime_combined')
    os.makedirs(os.path.join(lime_dir, 'correct'), exist_ok=True)
    os.makedirs(os.path.join(lime_dir, 'wrong'), exist_ok=True)

    # 重新创建 DataLoader
    full_dataset = test_loader.dataset.dataset
    sample_dataset = Subset(full_dataset, analysis_indices)
    sample_loader = DataLoader(sample_dataset, batch_size=1, shuffle=False)

    model.eval()

    def predict_fn(images):
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(device)
        # 针对 LIME 生成的样本进行标准化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        norm_images = (images - mean) / std

        with torch.no_grad():
            logits = model(norm_images)
        probas = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        return probas

    explainer = lime_image.LimeImageExplainer()

    pbar = tqdm(sample_loader, desc=f"生成 LIME 组合图 ({len(analysis_indices)} 样本)", unit="sample")

    # --- 轴刻度预计算 ---
    # 像素点位置 (0 到 IMAGE_SIZE-1)
    pixel_ticks = np.linspace(0, IMAGE_SIZE - 1, num=TICK_COUNT, dtype=int)

    # X 轴：时间 (秒)
    time_ticks_sec = pixel_ticks * (TOTAL_TIME_SEC / IMAGE_SIZE)
    time_labels = [f"{t:.2f}s" for t in time_ticks_sec]

    # Y 轴：频率 (kHz)
    max_freq = SAMPLE_RATE / 2 # 6000 Hz
    # 假设 Y=0 对应 Max_Freq，Y=223 对应 0 Hz
    freq_ticks_hz = max_freq * (1 - pixel_ticks / IMAGE_SIZE)
    freq_labels = [f"{f/1000:.1f}kHz" for f in freq_ticks_hz]
    # ---------------------

    for i, (input_tensor, true_label) in enumerate(pbar):

        global_index = analysis_indices[i]
        original_filename = get_original_filepath(test_loader.dataset, global_index)

        true_label = true_label.item()

        # 1. 明确计算模型的预测结果 (用于文件命名和LIME目标类)
        input_device = input_tensor.to(device)
        with torch.no_grad():
            output = model(input_device)
        model_predicted_class = output.argmax(dim=1).item()

        # 准备原图（去标准化）
        image_np = input_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        temp_img_denorm = denormalize_img(image_np) # 真实的去标准化原图

        try:
            explanation = explainer.explain_instance(
                temp_img_denorm, # LIME 最好使用去标准化后的图片作为输入
                predict_fn,
                top_labels=len(class_names), # 获取所有类别的权重
                hide_color=0,
                num_samples=1000,
            )
        except Exception as e:
            tqdm.write(f"警告: LIME 解释 {original_filename} 失败: {e}。")
            continue

        # 2. 使用模型的预测结果来获取 LIME 的特征权重，并作为可视化目标
        class_to_explain = model_predicted_class

        if class_to_explain not in explanation.local_exp:
            tqdm.write(f"警告: 预测类别 {class_names[class_to_explain]} 在 LIME 结果中没有权重。跳过。")
            continue

        feature_weights = dict(explanation.local_exp[class_to_explain])
        segments = explanation.segments # 获取 LIME 的图像分割图

        # --- 计算 LIME Overlay ---
        overlay = np.zeros((*temp_img_denorm.shape[:2], 4), dtype=np.float32)
        alpha_base = 0.6
        max_abs_weight = max((abs(w) for w in feature_weights.values()), default=1.0)
        if max_abs_weight == 0: max_abs_weight = 1.0

        for segment_idx in np.unique(segments):
            if segment_idx == 0: continue

            weight = feature_weights.get(segment_idx, 0)
            segment_pixels = segments == segment_idx

            color = [0, 0, 0]
            alpha = 0.0

            if weight > 0:
                # 正面贡献：绿色
                color = [0, 1, 0]
                alpha = alpha_base * (weight / max_abs_weight)
            elif weight < 0:
                # 负面贡献：红色
                color = [1, 0, 0]
                alpha = alpha_base * (abs(weight) / max_abs_weight)

            overlay[segment_pixels, :3] = color
            overlay[segment_pixels, 3] = alpha

        # --- 生成 1x2 组合图 (原图, LIME) ---
        fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5), dpi=100) # 调整尺寸

        # 1. 原图 (Spectrogram)
        axes[0].imshow(temp_img_denorm)
        axes[0].set_title("Original Spectrogram", fontsize=8)

        # 2. LIME Explanation
        axes[1].imshow(temp_img_denorm) # LIME 基图使用原图
        axes[1].imshow(overlay)          # 叠加 LIME Overlay
        axes[1].set_title("LIME Explanation", fontsize=8)

        # --- 应用时间和频率刻度 ---
        for ax in axes:
            ax.set_xticks(pixel_ticks)
            ax.set_xticklabels(time_labels, fontsize=6)
            ax.set_yticks(pixel_ticks)
            ax.set_yticklabels(freq_labels, fontsize=6)
            ax.set_xlabel("Time (s)", fontsize=7)
            ax.set_ylabel("Frequency (kHz)", fontsize=7)
            ax.grid(True, alpha=0.3)
            # ---------------------------

        class_name = class_names[true_label]
        pred_name = class_names[model_predicted_class]

        # 设定总标题并保存
        file_identifier = os.path.splitext(original_filename)[0]
        full_title = f"True: {class_name} | Pred: {pred_name} | File: {file_identifier}"
        fig.suptitle(full_title, fontsize=9)


        # 确定子文件夹
        folder = index_type_map.get(global_index, 'unknown')
        if folder == 'unknown': continue

        # 生成文件名：包含原始文件名、真值和预测
        img_filename = f"{file_identifier}_LIME_True_{class_name}_Pred_{pred_name}_Combined.png"
        img_path = os.path.join(lime_dir, folder, img_filename)

        plt.tight_layout(rect=[0, 0, 1, 0.95]) # 为 suptitle 留出空间
        plt.savefig(img_path)
        plt.close(fig)

    print(f"已保存 LIME 组合分析图到 {lime_dir}/")
