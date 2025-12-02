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
WINDOW_LENGTH = 1024                 # 小片段的滑窗长度
STEP_SIZE = 256                      # 小片段的滑窗步长
FRAME_STEP_SIZE = WINDOW_LENGTH      # 定义“大片”的步长 (1024)

# NFFT 决定 STFT 纵向维度 (频率轴)
NFFT_SIZE = 128
NOVERLAP_SIZE = NFFT_SIZE // 2       # 窗口重叠大小 (64)

# 目标子图尺寸 (用于计算图表大小，确保接近 224x224)
FINAL_IMAGE_SIZE = 224
DPI = 100
TARGET_SIZE_INCH = FINAL_IMAGE_SIZE / DPI # 2.24 英寸

# 每个文件子切片的数量
REQUIRED_SUB_SEGMENTS = 4

# --- 报告文件配置 (可配置) ---
REPORT_DIR = 'result'           # 报告文件根目录
PROJECT_NAME = 'cwru'                   # 项目名称 (可配置的 cwru 部分)
# 矢量图保存路径 (PDF 格式)
VECTOR_SAVE_PATH = pathlib.Path(REPORT_DIR) / PROJECT_NAME / "pre_data.pdf"


def normalize_signal(data):
    """ 对信号进行 Z-score 归一化 """
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std if std != 0 else data.astype(np.float64)


def generate_segment_data(signal, window_len, step_size, frame_step_size, nfft, noverlap):
    """
    对信号的第一个 FRAME_STEP_SIZE 长度进行切片，返回切片后的数据和STFT所需的轴数据。
    """
    import warnings
    # 抑制 Matplotlib UserWarning
    warnings.filterwarnings('ignore', category=UserWarning)

    signal = signal.flatten()
    data_len = len(signal)
    segment_data = []

    # 核心修改：只处理第一个大片段 (frame_start = 0)
    frame_start = 0

    if frame_start + window_len > data_len:
        warnings.resetwarnings()
        return [] # 信号太短

    segment_count = 0
    start = frame_start
    f = None
    t = None

    # 循环生成 REQUIRED_SUB_SEGMENTS (4) 个子切片
    while segment_count < REQUIRED_SUB_SEGMENTS and start + window_len <= data_len:

        # 确保子切片不会超出第一个 FRAME_STEP_SIZE (1024) 的范围
        if start >= frame_start + frame_step_size:
            break

        end = start + window_len
        windowed_data = signal[start:end]

        # --- STFT 轴数据计算 (仅计算轴，不计算矩阵) ---
        normalized_data = normalize_signal(windowed_data)
        if f is None:
            # 只计算一次 STFT 以获取 f 和 t 轴，使用第一个窗口数据
            f, t, _ = stft(
                normalized_data, fs=12000, nperseg=nfft, noverlap=noverlap, nfft=nfft, return_onesided=True, boundary=None
            )

        # 存储 (原始数据, STFT f 轴, STFT t 轴)
        segment_data.append((windowed_data, f, t))

        segment_count += 1
        # 移动到下一个子切片的起始位置
        start += step_size

    warnings.resetwarnings()
    return segment_data


def process_cwru_dataset():
    """ 遍历原始数据集，只处理每个子目录中的第一个 .mat 文件，并在 N x 8 宫格中展示所有结果，并保存为 PDF 矢量图。 """
    print(f"--- 原始数据根目录: {ROOT_DIR} ---")

    # 禁用 Matplotlib 交互模式，以便在最后统一绘图
    plt.ioff()

    all_tasks = []

    # 收集每个目录下的第一个 .mat 文件
    for root, dirs, files in os.walk(ROOT_DIR):
        mat_files = sorted([f for f in files if f.endswith('.mat')])

        if mat_files:
            first_file_name = mat_files[0]
            mat_path = pathlib.Path(root) / first_file_name
            all_tasks.append((pathlib.Path(root), mat_path))

    if not all_tasks:
        print(f"错误：在 {ROOT_DIR} 中未找到任何 .mat 文件。请检查路径和结构。")
        return

    # 用于收集所有切片数据和标签的列表
    all_segment_data = [] # 存储 (windowed_data, f, t)
    category_labels = []  # <<< 重新启用类别标签收集

    pbar_data = tqdm(all_tasks, desc="文件处理进度 (每类别第一个)", unit="file")

    # --- 步骤 1: 处理所有文件并收集数据 ---
    for current_folder, mat_path in pbar_data:
        filename = mat_path.name

        # 只保留相对目录路径作为标签
        relative_path = os.path.relpath(current_folder, ROOT_DIR)
        category_label = relative_path if relative_path != '.' else 'Root' # 防止根目录标签为空

        try:
            mat_data = scipy.io.loadmat(mat_path)
            data_key = None
            # 查找信号键 (省略查找逻辑，假设 data_key 找到)
            for key in mat_data.keys():
                if 'DE_time' in key or 'FE_time' in key:
                    data_key = key
                    break
            if not data_key:
                # 尝试其他键
                for key in mat_data.keys():
                    if 'X' in key or 'Y' in key:
                        data_key = key
                        break

            if not data_key:
                tqdm.write(f"!!! 警告: 文件 {filename} 中未找到可用信号键。跳过。")
                continue

            raw_signal = mat_data[data_key]
            signal_data = raw_signal[:, 0] if raw_signal.ndim > 1 else raw_signal

            # 生成切片数据
            segment_data = generate_segment_data(
                signal_data, WINDOW_LENGTH, STEP_SIZE, FRAME_STEP_SIZE, NFFT_SIZE, NOVERLAP_SIZE
            )

            if len(segment_data) == REQUIRED_SUB_SEGMENTS:
                # 存储数据和标签
                all_segment_data.extend(segment_data)
                # 为每个子切片存储对应的类别标签
                for i in range(REQUIRED_SUB_SEGMENTS):
                    category_labels.append(category_label)
            else:
                tqdm.write(f"!!! 警告: 文件 {filename} 信号太短。只生成了 {len(segment_data)} 个子切片。跳过。")

        except Exception as e:
            tqdm.write(f"!!! 错误: 处理文件 {filename} 时发生错误: {e}")

    # --- 步骤 2: 在 N x 8 宫格中绘制所有收集到的数据 ---
    N = len(category_labels) // REQUIRED_SUB_SEGMENTS if REQUIRED_SUB_SEGMENTS > 0 else 0 # 成功处理的类别数

    if N == 0:
        print("\n--- 没有数据可供绘图。 ---")
        return

    ROWS = N
    COLS = REQUIRED_SUB_SEGMENTS * 2 # 8 列 (4 * 频谱图 + 4 * 波形图)

    # *** 计算 figsize ***
    # 保持宽度略大，因为 Y 轴标题变长了
    PLOT_WIDTH_INCH = COLS * TARGET_SIZE_INCH + 2.5
    PLOT_HEIGHT_INCH = ROWS * TARGET_SIZE_INCH + 2.0

    print(f"\n--- 正在 {ROWS}x{COLS} 宫格中绘制 N={ROWS} 个类别 (目标子图像素: {FINAL_IMAGE_SIZE}x{FINAL_IMAGE_SIZE})... ---")

    fig, axes = plt.subplots(nrows=ROWS, ncols=COLS, figsize=(PLOT_WIDTH_INCH, PLOT_HEIGHT_INCH), squeeze=False)
    fig.set_dpi(DPI)

    # *** 图表标题使用英文 (防止中文) ***
    fig.suptitle(f"STFT Spectrograms and Overlap-Coded Waveforms (N={ROWS} Categories)", fontsize=16)

    # 预计算所有 STFT 矩阵并找到全局 vmax
    global_vmax = 0.0
    stft_list = []

    # 使用 tqdm 包装 STFT 预计算循环
    pbar_stft = tqdm(all_segment_data, desc="预计算 STFT", unit="segment")
    for windowed_data, _, _ in pbar_stft:
        normalized_data = normalize_signal(windowed_data)
        _, _, Zxx = stft(
            normalized_data, fs=12000, nperseg=NFFT_SIZE, noverlap=NOVERLAP_SIZE, nfft=NFFT_SIZE, return_onesided=True, boundary=None
        )
        magnitude_spectrum = np.abs(Zxx)
        stft_list.append(magnitude_spectrum)
        if np.max(magnitude_spectrum) > global_vmax:
            global_vmax = np.max(magnitude_spectrum)

    im = None # 用于存储可映射对象以便创建 colorbar

    # 第三次循环：绘制图表
    # 使用 tqdm 包装绘图循环
    pbar_plot = tqdm(range(len(all_segment_data)), desc="绘制图表", unit="segment")
    for i in pbar_plot:
        row = i // REQUIRED_SUB_SEGMENTS
        col_offset = i % REQUIRED_SUB_SEGMENTS

        # 频谱图在奇数列 (0, 2, 4, 6)
        ax_stft = axes[row, col_offset * 2]
        # 波形图在偶数列 (1, 3, 5, 7)
        ax_wave = axes[row, col_offset * 2 + 1]

        windowed_data, f, t = all_segment_data[i]
        magnitude_spectrum = stft_list[i]

        # 获取当前子切片对应的类别标签
        current_category = category_labels[i]

        # --- 1. 绘制 STFT 频谱图 (奇数列) ---
        im = ax_stft.pcolormesh(t, f, magnitude_spectrum, shading='gouraud', cmap='viridis', vmax=global_vmax)

        # 设置频谱图标签 (英文)
        ax_stft.set_title(f"Spectrogram {col_offset+1}", fontsize=8)
        ax_stft.set_xlabel("Time (s)", fontsize=9)

        # *** 关键修改：Y 轴标题加上类别名称 ***
        y_label_stft = f"Frequency (Hz)({current_category})"
        ax_stft.set_ylabel(y_label_stft, fontsize=9)
        # 所有频谱图都显示 Y 轴刻度
        ax_stft.tick_params(axis='y', labelleft=True)

        # ------------------------------------

        # --- 2. 绘制波形图 (偶数列) ---

        overlap_len = WINDOW_LENGTH - STEP_SIZE # 768
        wave_x = np.arange(WINDOW_LENGTH)

        # 绘制重叠部分 (前 768 个点) -> 红色
        ax_wave.plot(wave_x[:overlap_len], windowed_data[:overlap_len], color='red', linewidth=1, zorder=1)

        # 绘制非重叠部分 (后 256 个点) -> 黑色
        ax_wave.plot(wave_x[overlap_len:], windowed_data[overlap_len:], color='black', linewidth=1, zorder=2)

        # 绘制第一个切片的前 768 个点为黑色（因为它是整个数据的开始，没有前一段重叠）
        if col_offset == 0:
            ax_wave.plot(wave_x[:overlap_len], windowed_data[:overlap_len], color='black', linewidth=1, zorder=2)


        # 设置波形图标签 (英文)
        ax_wave.set_title(f"Waveform {col_offset+1} (Red=Overlap)", fontsize=8)
        ax_wave.set_xlabel("Samples", fontsize=9)

        # *** 关键修改：Y 轴标题加上类别名称 ***
        y_label_wave = f"Amplitude ({current_category})"
        ax_wave.set_ylabel(y_label_wave, fontsize=9)
        # 所有波形图都显示 Y 轴刻度
        ax_wave.tick_params(axis='y', labelleft=True)

        # *** 关键：所有子图都显示 X 轴刻度 (保持不变) ***


    # 调整布局并添加一个统一的 colorbar
    # *** 关键修改：调整左侧边距以容纳更长的 Y 轴标题 ***
    fig.tight_layout(rect=[0.05, 0.03, 0.95, 0.97], w_pad=0.5)

    if im:
        cbar_ax = fig.add_axes([0.96, 0.1, 0.007, 0.8])
        fig.colorbar(im, cax=cbar_ax, label='Magnitude')

    # --- 关键：保存为 PDF 矢量图 ---
    try:
        # 确保目标目录存在
        VECTOR_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        # 保存为 PDF 格式
        fig.savefig(VECTOR_SAVE_PATH, format='pdf')
        print(f"\n--- 矢量图已保存至: {VECTOR_SAVE_PATH.resolve()} ---")
    except Exception as e:
        print(f"\n!!! 警告: 保存 PDF 文件失败: {e}")

    print("\n--- 所有选定文件处理完成 ---")
    print(f"成功处理并展示了 {ROWS} 个类别 ({ROWS * REQUIRED_SUB_SEGMENTS * 2} 个子图)。")


if __name__ == '__main__':
    # 确保在开始处理前禁用 Matplotlib 自动弹出窗口
    plt.ioff()
    if not os.path.exists(ROOT_DIR):
        print(f"错误：找不到原始数据目录: {ROOT_DIR}")
    else:
        process_cwru_dataset()
