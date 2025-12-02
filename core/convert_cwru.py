import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import pathlib
from tqdm import tqdm
import re

# --- 配置参数 ---
ROOT_DIR = './datasets/data_cwru'    # 原始数据根目录
TARGET_DIR = './data/data_cwru' # 目标图片存储根目录
WINDOW_LENGTH = 1024                 # 小片段的滑窗长度
STEP_SIZE = 256                      # 小片段的滑窗步长
FRAME_STEP_SIZE = WINDOW_LENGTH      # 定义“大片”的步长

# NFFT 决定 STFT 纵向维度 (频率轴)
NFFT_SIZE = 128
NOVERLAP_SIZE = NFFT_SIZE // 2       # 窗口重叠大小 (64)

# 目标图片尺寸
FINAL_IMAGE_SIZE = 224 # 最终保存的图片尺寸为 224x224 像素

# 绘图配置：使用固定的 DPI 和对应的英寸大小来实现目标像素尺寸
# 224 像素 / 100 DPI = 2.24 英寸
DPI = 100
TARGET_SIZE_INCH = FINAL_IMAGE_SIZE / DPI


def normalize_signal(data):
    """ 对信号进行 Z-score 归一化 """
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std if std != 0 else data


def generate_images_with_sliding_prefix(signal, window_len, step_size, frame_step_size, nfft, noverlap, target_folder, initial_frame_id, final_size_inch, dpi):
    """
    对信号进行滑窗、STFT转换并保存为 224x224 图像。
    """
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)

    signal = signal.flatten()
    data_len = len(signal)

    target_folder.mkdir(parents=True, exist_ok=True)

    current_frame_id = initial_frame_id
    total_samples_generated = 0

    # 循环滑窗
    for frame_start in range(0, data_len, frame_step_size):
        if frame_start + window_len > data_len:
            break

        frame_prefix = f"F{current_frame_id:04d}"
        segment_count = 0
        start = frame_start

        while start + window_len <= data_len:
            end = start + window_len
            windowed_data = signal[start:end]

            # --- STFT 过程 ---
            normalized_data = normalize_signal(windowed_data)
            # 假设采样频率 fs=12000 Hz
            f, t, Zxx = stft(
                normalized_data, fs=12000, nperseg=nfft, noverlap=noverlap, nfft=nfft, return_onesided=True, boundary=None
            )
            magnitude_spectrum = np.abs(Zxx)

            # --- 绘图和保存：强制 224x224 像素 ---
            # 设置画布尺寸和 DPI，确保输出像素为 FINAL_IMAGE_SIZE x FINAL_IMAGE_SIZE
            plt.figure(figsize=(final_size_inch, final_size_inch), dpi=dpi)

            # 绘制频谱图
            # pcolormesh 的输入尺寸是 65x15，Matplotlib 会自动拉伸到 224x224
            plt.pcolormesh(t, f, magnitude_spectrum, shading='gouraud', cmap='viridis')

            # 移除所有边框和坐标轴，让频谱图紧密填充整个画布
            plt.axis('off')
            plt.gca().set_position([0, 0, 1, 1])

            file_name = f"{frame_prefix}_{segment_count:04d}.png"
            save_path = target_folder / file_name

            # 使用 bbox_inches='tight' 和 pad_inches=0 裁剪到只有数据区域
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            segment_count += 1
            total_samples_generated += 1
            start += step_size

            if start >= frame_start + frame_step_size:
                break

        current_frame_id += 1

    warnings.resetwarnings()
    return total_samples_generated, current_frame_id


def process_cwru_dataset():
    """ 遍历原始数据集并进行处理 """
    print(f"--- 目标根目录: {ROOT_DIR} ---")

    all_tasks = []
    for root, dirs, files in os.walk(ROOT_DIR):
        mat_files = [pathlib.Path(root) / f for f in files if f.endswith('.mat')]
        if mat_files:
            all_tasks.extend([(pathlib.Path(root), f) for f in mat_files])

    if not all_tasks:
        print(f"错误：在 {ROOT_DIR} 中未找到任何 .mat 文件。请检查路径和结构。")
        return

    # 计算 STFT 原始尺寸（仅用于日志输出）
    original_height = NFFT_SIZE // 2 + 1 # 65
    original_width = int(np.floor((WINDOW_LENGTH - NFFT_SIZE) / (NFFT_SIZE - NOVERLAP_SIZE))) + 1 # 15

    global_total_processed_samples = 0
    pbar = tqdm(all_tasks, desc="总文件处理进度", unit="file")

    for current_folder, mat_path in pbar:
        filename = mat_path.name
        relative_path = os.path.relpath(current_folder, ROOT_DIR)
        target_class_folder = pathlib.Path(TARGET_DIR) / relative_path
        initial_frame_id = 1
        pbar.set_description(f"处理中: {relative_path}/{filename}")

        try:
            mat_data = scipy.io.loadmat(mat_path)
            data_key = None

            # 查找信号键 (保留原始查找逻辑)
            for key in mat_data.keys():
                if 'DE_time' in key or 'FE_time' in key:
                    data_key = key
                    break
            if not data_key:
                for key in mat_data.keys():
                    if 'X' in key or 'Y' in key:
                        data_key = key
                        break

            if not data_key:
                tqdm.write(f"!!! 警告: 文件 {filename} 中未找到可用信号键。跳过。")
                continue

            raw_signal = mat_data[data_key]
            signal_data = raw_signal[:, 0] if raw_signal.ndim > 1 else raw_signal

            # 调用生成函数，传入 224x224 尺寸参数
            samples_generated, next_frame_id = generate_images_with_sliding_prefix(
                signal_data,
                WINDOW_LENGTH,
                STEP_SIZE,
                FRAME_STEP_SIZE,
                NFFT_SIZE,
                NOVERLAP_SIZE,
                target_class_folder,
                initial_frame_id,
                TARGET_SIZE_INCH, # 2.24 英寸
                DPI               # 100 DPI
            )

            global_total_processed_samples += samples_generated
            pbar.set_postfix_str(f"样本总数: {global_total_processed_samples:,}")

        except Exception as e:
            tqdm.write(f"!!! 错误: 处理文件 {filename} 时发生错误: {e}")

    print("\n--- 所有文件处理完成 ---")
    print(f"最终生成图片样本总数: {global_total_processed_samples:,} 张")
    print(f"图片尺寸已统一为: {FINAL_IMAGE_SIZE}x{FINAL_IMAGE_SIZE} 像素 (原始STFT尺寸为 {original_width}x{original_height})")


if __name__ == '__main__':
    # 禁用 Matplotlib 自动弹出窗口 (防止服务器环境报错)
    plt.ioff()
    if not os.path.exists(ROOT_DIR):
        print(f"错误：找不到原始数据目录: {ROOT_DIR}")
    else:
        process_cwru_dataset()
