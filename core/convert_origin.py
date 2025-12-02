import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import pathlib
from tqdm import tqdm
from nptdms import TdmsFile # 引入 TDMS 文件读取库

# --- 配置参数 ---
ROOT_DIR = './datasets/data_origin'    # 原始 TDMS 数据根目录
TARGET_DIR = './data/data_origin' # 目标图片存储根目录
WINDOW_LENGTH = 1024                 # 小片段的滑窗长度
STEP_SIZE = 256                      # 小片段的滑窗步长
FRAME_STEP_SIZE = WINDOW_LENGTH      # 定义“大片”的步长

# NFFT 决定 STFT 纵向维度 (频率轴)
NFFT_SIZE = 128
NOVERLAP_SIZE = NFFT_SIZE // 2       # 窗口重叠大小 (64)

# 目标图片尺寸
FINAL_IMAGE_SIZE = 224 # 最终保存的图片尺寸为 224x224 像素

# 绘图配置
DPI = 100
TARGET_SIZE_INCH = FINAL_IMAGE_SIZE / DPI

# --- TDMS 文件专用配置 ---
# 固定读取您之前确认的 RZ 通道
TARGET_GROUP_NAME = '数据'
TARGET_CHANNEL_NAME = 'RZ'

def normalize_signal(data):
    """ 对信号进行 Z-score 归一化 """
    mean = np.mean(data)
    std = np.std(data)
    # 使用 np.float64 确保后续计算精度，同时避免整型溢出
    return (data - mean) / std if std != 0 else data.astype(np.float64)


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

    fs_sample = 12000

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

            f, t, Zxx = stft(
                normalized_data, fs=fs_sample, nperseg=nfft, noverlap=noverlap, nfft=nfft, return_onesided=True, boundary=None
            )
            magnitude_spectrum = np.abs(Zxx)

            # --- 绘图和保存：强制 224x224 像素 ---
            plt.figure(figsize=(final_size_inch, final_size_inch), dpi=dpi)
            plt.pcolormesh(t, f, magnitude_spectrum, shading='gouraud', cmap='viridis')
            plt.axis('off')
            plt.gca().set_position([0, 0, 1, 1])

            file_name = f"{frame_prefix}_{segment_count:04d}.png"
            save_path = target_folder / file_name

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


def process_tdms_dataset():
    """ 遍历 TDMS 数据集并固定读取 RZ 通道 """
    print(f"--- 目标 TDMS 数据根目录: {ROOT_DIR} ---")
    print(f"--- 将固定读取通道: {TARGET_GROUP_NAME}/{TARGET_CHANNEL_NAME} ---")

    # 查找所有 .tdms 文件
    all_tasks = []
    for root, dirs, files in os.walk(ROOT_DIR):
        tdms_files = [pathlib.Path(root) / f for f in files if f.endswith('.tdms')]
        if tdms_files:
            all_tasks.extend([(pathlib.Path(root), f) for f in tdms_files])

    if not all_tasks:
        print(f"错误：在 {ROOT_DIR} 中未找到任何 .tdms 文件。请检查路径和结构。")
        return

    original_height = NFFT_SIZE // 2 + 1
    original_width = int(np.floor((WINDOW_LENGTH - NFFT_SIZE) / (NFFT_SIZE - NOVERLAP_SIZE))) + 1

    global_total_processed_samples = 0
    pbar = tqdm(all_tasks, desc="总文件处理进度", unit="file")

    for current_folder, tdms_path in pbar:
        filename = tdms_path.name
        relative_path = os.path.relpath(current_folder, ROOT_DIR)
        target_class_folder = pathlib.Path(TARGET_DIR) / relative_path
        initial_frame_id = 1
        pbar.set_description(f"处理中: {relative_path}/{filename}")

        try:
            # --- TDMS 文件读取逻辑 ---
            with TdmsFile.open(tdms_path) as tdms_file:

                # 检查组和通道是否存在
                if TARGET_GROUP_NAME not in tdms_file or TARGET_CHANNEL_NAME not in tdms_file[TARGET_GROUP_NAME]:
                    tqdm.write(f"!!! 警告: 文件 {filename} 中未找到通道 '{TARGET_GROUP_NAME}/{TARGET_CHANNEL_NAME}'。跳过。")
                    continue

                # 读取 RZ 通道数据
                raw_signal = tdms_file[TARGET_GROUP_NAME][TARGET_CHANNEL_NAME].read_data()

            # 由于 TDMS 读取的数据已经是 NumPy 数组，无需再进行维度处理
            signal_data = raw_signal

            # 调用生成函数
            samples_generated, next_frame_id = generate_images_with_sliding_prefix(
                signal_data,
                WINDOW_LENGTH,
                STEP_SIZE,
                FRAME_STEP_SIZE,
                NFFT_SIZE,
                NOVERLAP_SIZE,
                target_class_folder,
                initial_frame_id,
                TARGET_SIZE_INCH,
                DPI
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
        process_tdms_dataset()
