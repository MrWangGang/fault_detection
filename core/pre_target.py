import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import pathlib
from tqdm import tqdm
from nptdms import TdmsFile # 引入 TDMS 文件读取库

# --- 配置参数 ---
ROOT_DIR = './datasets/data_target'    # 原始 TDMS 数据根目录
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

# 每个文件子切片的数量 (用于宫格图展示)
REQUIRED_SUB_SEGMENTS = 4

# --- TDMS 文件专用配置 ---
TARGET_GROUP_NAME = '数据'
TARGET_CHANNEL_NAME = 'RZ'

# --- 报告文件配置 ---
REPORT_DIR = 'result'           # 报告文件根目录
PROJECT_NAME = 'target'          # 项目名称
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
    这个函数只提取前 REQUIRED_SUB_SEGMENTS 个子切片。
    """
    import warnings
    # 抑制 Matplotlib UserWarning
    warnings.filterwarnings('ignore', category=UserWarning)

    signal = signal.flatten()
    data_len = len(signal)
    segment_data = []

    fs_sample = 12000

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
                normalized_data, fs=fs_sample, nperseg=nfft, noverlap=noverlap, nfft=nfft, return_onesided=True, boundary=None
            )

        # 存储 (原始数据, STFT f 轴, STFT t 轴)
        segment_data.append((windowed_data, f, t))

        segment_count += 1
        # 移动到下一个子切片的起始位置
        start += step_size

    warnings.resetwarnings()
    return segment_data


def process_tdms_dataset_for_overview():
    """ 遍历 TDMS 数据集，只处理每个子目录下的第一个 .tdms 文件，用于绘制 N x 8 宫格图。 """
    print(f"--- 原始 TDMS 数据根目录: {ROOT_DIR} ---")
    print(f"--- 将固定读取通道: {TARGET_GROUP_NAME}/{TARGET_CHANNEL_NAME} ---")

    plt.ioff() # 禁用 Matplotlib 交互模式

    all_tasks = []

    # 收集每个目录下的第一个 .tdms 文件
    for root, dirs, files in os.walk(ROOT_DIR):
        tdms_files = sorted([f for f in files if f.endswith('.tdms')])

        if tdms_files:
            # 只取第一个文件
            first_file_name = tdms_files[0]
            tdms_path = pathlib.Path(root) / first_file_name
            # 存储 (当前目录, 文件路径)
            all_tasks.append((pathlib.Path(root), tdms_path))

    if not all_tasks:
        print(f"错误：在 {ROOT_DIR} 中未找到任何 .tdms 文件。请检查路径和结构。")
        return

    # 用于收集所有切片数据和类别标签的列表
    all_segment_data = [] # 存储 (windowed_data, f, t)
    category_labels = []  # <<< 重新启用类别标签收集

    pbar_data = tqdm(all_tasks, desc="文件处理进度 (每类别第一个)", unit="file")

    # --- 步骤 1: 处理所有文件并收集数据 ---
    for current_folder, tdms_path in pbar_data:
        filename = tdms_path.name

        # 类别标签：只保留相对目录路径，并去除 ROOT_DIR
        relative_path = os.path.relpath(current_folder, ROOT_DIR)
        category_label = relative_path if relative_path != '.' else 'Root' # 防止根目录标签为空

        try:
            # *** TDMS 文件读取逻辑 ***
            with TdmsFile.open(tdms_path) as tdms_file:
                if TARGET_GROUP_NAME not in tdms_file or TARGET_CHANNEL_NAME not in tdms_file[TARGET_GROUP_NAME]:
                    tqdm.write(f"!!! 警告: 文件 {filename} 中未找到通道 '{TARGET_GROUP_NAME}/{TARGET_CHANNEL_NAME}'。跳过。")
                    continue

                raw_signal = tdms_file[TARGET_GROUP_NAME][TARGET_CHANNEL_NAME].read_data()

            signal_data = raw_signal

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
                tqdm.write(f"!!! 警告: 文件 {filename} 信号太短或数据不足 {REQUIRED_SUB_SEGMENTS} 个切片。跳过。")

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

    fig.suptitle(f"STFT Spectrograms and Overlap-Coded Waveforms (N={ROWS} Categories)", fontsize=16)

    # 预计算所有 STFT 矩阵并找到全局 vmax
    global_vmax = 0.0
    stft_list = []
    fs_sample = 12000

    pbar_stft = tqdm(all_segment_data, desc="预计算 STFT", unit="segment")
    for windowed_data, _, _ in pbar_stft:
        normalized_data = normalize_signal(windowed_data)
        _, _, Zxx = stft(
            normalized_data, fs=fs_sample, nperseg=NFFT_SIZE, noverlap=NOVERLAP_SIZE, nfft=NFFT_SIZE, return_onesided=True, boundary=None
        )
        magnitude_spectrum = np.abs(Zxx)
        stft_list.append(magnitude_spectrum)
        if np.max(magnitude_spectrum) > global_vmax:
            global_vmax = np.max(magnitude_spectrum)

    im = None

    # 第三次循环：绘制图表
    pbar_plot = tqdm(range(len(all_segment_data)), desc="绘制图表", unit="segment")
    for i in pbar_plot:
        row = i // REQUIRED_SUB_SEGMENTS
        col_offset = i % REQUIRED_SUB_SEGMENTS

        ax_stft = axes[row, col_offset * 2]
        ax_wave = axes[row, col_offset * 2 + 1]

        windowed_data, f, t = all_segment_data[i]
        magnitude_spectrum = stft_list[i]

        # 获取当前子切片对应的类别标签
        current_category = category_labels[i]

        # --- 1. 绘制 STFT 频谱图 (奇数列) ---
        im = ax_stft.pcolormesh(t, f, magnitude_spectrum, shading='gouraud', cmap='viridis', vmax=global_vmax)

        ax_stft.set_title(f"Spectrogram {col_offset+1}", fontsize=8)
        ax_stft.set_xlabel("Time (s)", fontsize=9)

        # *** 关键修改：Y 轴标题加上类别名称 ***
        y_label_stft = f"Frequency (Hz)({current_category})"
        ax_stft.set_ylabel(y_label_stft, fontsize=9)
        ax_stft.tick_params(axis='y', labelleft=True)

        # ------------------------------------

        # --- 2. 绘制波形图 (偶数列) ---

        overlap_len = WINDOW_LENGTH - STEP_SIZE # 768
        wave_x = np.arange(WINDOW_LENGTH)

        # 绘制重叠部分 (前 768 个点) -> 红色
        ax_wave.plot(wave_x[:overlap_len], windowed_data[:overlap_len], color='red', linewidth=1, zorder=1)

        # 绘制非重叠部分 (后 256 个点) -> 黑色
        ax_wave.plot(wave_x[overlap_len:], windowed_data[overlap_len:], color='black', linewidth=1, zorder=2)

        # 绘制第一个切片的前 768 个点为黑色
        if col_offset == 0:
            ax_wave.plot(wave_x[:overlap_len], windowed_data[:overlap_len], color='black', linewidth=1, zorder=2)


        ax_wave.set_title(f"Waveform {col_offset+1} (Red=Overlap)", fontsize=8)
        ax_wave.set_xlabel("Samples", fontsize=9)

        # *** 关键修改：Y 轴标题加上类别名称 ***
        y_label_wave = f"Amplitude ({current_category})"
        ax_wave.set_ylabel(y_label_wave, fontsize=9)
        ax_wave.tick_params(axis='y', labelleft=True)

        # ------------------------------------


    # 调整布局并添加一个统一的 colorbar
    # *** 关键修改：调整左侧边距以容纳更长的 Y 轴标题 ***
    fig.tight_layout(rect=[0.05, 0.03, 0.95, 0.97], w_pad=0.5)

    if im:
        cbar_ax = fig.add_axes([0.96, 0.1, 0.007, 0.8])
        fig.colorbar(im, cax=cbar_ax, label='Magnitude')

    # --- 关键：保存为 PDF 矢量图 ---
    try:
        VECTOR_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(VECTOR_SAVE_PATH, format='pdf')
        print(f"\n--- 矢量图已保存至: {VECTOR_SAVE_PATH.resolve()} ---")
    except Exception as e:
        print(f"\n!!! 警告: 保存 PDF 文件失败: {e}")

    print("\n--- 所有选定 TDMS 文件处理完成 ---")
    print(f"成功处理并展示了 {N} 个类别 ({N * REQUIRED_SUB_SEGMENTS * 2} 个子图)。")


if __name__ == '__main__':
    plt.ioff()
    if not os.path.exists(ROOT_DIR):
        print(f"错误：找不到原始数据目录: {ROOT_DIR}")
    else:
        process_tdms_dataset_for_overview()
