# data_provider/eeg_processor.py
"""
EEG数据处理器 - 包含完整的EEG处理逻辑
"""
import os
import mne
import numpy as np
import pandas as pd
import torch
import json
from collections import defaultdict, Counter
import re
import warnings
warnings.filterwarnings('ignore')

# 检查scipy是否可用
try:
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠ 警告: scipy 未安装, 无法使用下采样功能")

def load_text_maps(json_path):
    """加载textmaps.json映射文件"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            text_maps = json.load(f)
        print(f"✓ 成功加载textmaps.json，包含 {len(text_maps)} 个映射")
        return text_maps
    except Exception as e:
        print(f"❌ 加载textmaps.json失败: {e}")
        return None

def find_imagine_fif_files(data_dir):
    """查找imagine任务的FIF文件"""
    fif_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.fif', '.fif.gz')) and 'imagine' in file.lower():
                fif_files.append(os.path.join(root, file))
    return fif_files

def extract_metadata_from_epochs(epochs):
    """从epochs对象中提取Metadata信息"""
    if hasattr(epochs, 'metadata') and epochs.metadata is not None:
        if 'Word' in epochs.metadata.columns:
            return epochs.metadata
    return None

def extract_word_labels_from_metadata(epochs, n_epochs):
    """从Metadata中提取Word文字标签"""
    metadata = extract_metadata_from_epochs(epochs)
    if metadata is None or 'Word' not in metadata.columns:
        return ["unknown"] * n_epochs
    
    trial_labels = []
    for i in range(min(n_epochs, len(metadata))):
        word_value = metadata.iloc[i]['Word']
        if pd.isna(word_value):
            word_label = "unknown"
        else:
            word_label = str(word_value).strip()
        trial_labels.append(word_label)
    
    while len(trial_labels) < n_epochs:
        trial_labels.append(trial_labels[-1] if trial_labels else "unknown")
    
    return trial_labels


def simple_downsample(data, factor, axis=1, method='mean'):
    """
    简单的下采样实现
    Args:
        data: 输入数据
        factor: 下采样因子
        axis: 时间轴
        method: 下采样方法, 'mean' 或 'decimate'
    
    Returns:
        下采样后的数据
    """
    if factor <= 1:
        return data
    
    n_samples = data.shape[axis]
    new_n = n_samples // factor
    if new_n <= 0:
        new_n = 1
    
    if method == 'mean':
        # 平均下采样
        if axis == 0:
            # 重塑为 (factor, new_n, ...) 然后平均
            new_shape = (new_n, factor) + data.shape[1:]
            reshaped = data[:new_n*factor].reshape(new_shape)
            return reshaped.mean(axis=1)
        elif axis == 1:
            # 重塑为 (..., factor, new_n) 然后平均
            new_shape = (data.shape[0], new_n, factor) + data.shape[2:]
            reshaped = data[:, :new_n*factor].reshape(new_shape)
            return reshaped.mean(axis=2)
        elif axis == 2:
            new_shape = (data.shape[0], data.shape[1], new_n, factor)
            reshaped = data[:, :, :new_n*factor].reshape(new_shape)
            return reshaped.mean(axis=3)
    else:
        # 简单抽取
        idx = np.arange(0, n_samples, factor, dtype=int)
        if idx.shape[0] == 0:
            idx = np.array([0])
        
        if axis == 0:
            return data[idx]
        elif axis == 1:
            return data[:, idx]
        elif axis == 2:
            return data[:, :, idx]
    
    return data

def calculate_required_timepoints(original_fs, target_fs, duration_seconds=None, original_timepoints=None):
    """
    根据采样率和时长计算所需的时间点数
    
    Args:
        original_fs: 原始采样率
        target_fs: 目标采样率
        duration_seconds: 时长（秒）
        original_timepoints: 原始时间点数
    
    Returns:
        目标时间点数
    """
    if duration_seconds is not None:
        # 通过时长计算
        target_timepoints = int(duration_seconds * target_fs)
    elif original_timepoints is not None:
        # 通过原始时间点计算
        duration_seconds = original_timepoints / original_fs
        target_timepoints = int(duration_seconds * target_fs)
    else:
        # 默认3秒
        target_timepoints = int(3.0 * target_fs)
    
    return target_timepoints
'''def apply_antialiasing_filter(data, original_fs, target_fs, axis=1, debug=False):
    """
    应用抗混叠滤波
    
    注意：对于500Hz -> 256Hz的下采样，滤波器截止频率为102.4Hz
    由于采样率很高，截止频率也在合理范围内
    """
    if not HAS_SCIPY:
        if debug:
            print(f"  [apply_antialiasing_filter] scipy 未安装，跳过滤波")
        return data
    
    if target_fs >= original_fs:
        if debug:
            print(f"  [apply_antialiasing_filter] 目标采样率{target_fs}Hz >= 原始采样率{original_fs}Hz，跳过滤波")
        return data
    
    # 对于500Hz -> 256Hz
    # 原始Nyquist: 250Hz
    # 目标Nyquist: 128Hz
    # 截止频率: 128Hz * 0.8 = 102.4Hz
    # 这是完全合理的
    try:
        # 计算滤波器参数
        nyquist_freq = original_fs / 2.0
        cutoff_freq = target_fs / 2.0 * 0.8  # 留20%余量
        
        if debug:
            print(f"  [apply_antialiasing_filter] Nyquist频率: {nyquist_freq}Hz")
            print(f"  [apply_antialiasing_filter] 截止频率: {cutoff_freq}Hz")
            print(f"  [apply_antialiasing_filter] 归一化截止频率: {cutoff_freq/nyquist_freq}")
        
        # 验证截止频率
        if cutoff_freq <= 0.1 or cutoff_freq >= nyquist_freq * 0.99:
            if debug:
                print(f"  [apply_antialiasing_filter] 警告: 截止频率{cutoff_freq}Hz超出范围，使用简化下采样")
            return data
        
        # 设计Butterworth低通滤波器
        sos = signal.butter(4, cutoff_freq/nyquist_freq, btype='low', output='sos')
        
        if debug:
            print(f"  [apply_antialiasing_filter] 应用滤波器...")
        
        # 应用滤波
        filtered = signal.sosfiltfilt(sos, data, axis=axis)
        
        if debug:
            print(f"  [apply_antialiasing_filter] 滤波完成")
        
        return filtered
        
    except Exception as e:
        if debug:
            print(f"  [apply_antialiasing_filter] 滤波失败: {e}")
            print(f"  [apply_antialiasing_filter] 返回原始数据")
        return data'''


def simple_downsample(data, factor, axis=1, method='mean'):
    """
    简单的下采样实现
    
    Args:
        data: 输入数据
        factor: 下采样因子
        axis: 时间轴
        method: 下采样方法, 'mean' 或 'decimate'
    
    Returns:
        下采样后的数据
    """
    if factor <= 1:
        return data
    
    n_samples = data.shape[axis]
    new_n = n_samples // factor
    if new_n <= 0:
        new_n = 1
    
    if method == 'mean':
        # 平均下采样
        if axis == 0:
            # 重塑为 (factor, new_n, ...) 然后平均
            new_shape = (new_n, factor) + data.shape[1:]
            reshaped = data[:new_n*factor].reshape(new_shape)
            return reshaped.mean(axis=1)
        elif axis == 1:
            # 重塑为 (..., factor, new_n) 然后平均
            new_shape = (data.shape[0], new_n, factor) + data.shape[2:]
            reshaped = data[:, :new_n*factor].reshape(new_shape)
            return reshaped.mean(axis=2)
        elif axis == 2:
            new_shape = (data.shape[0], data.shape[1], new_n, factor)
            reshaped = data[:, :, :new_n*factor].reshape(new_shape)
            return reshaped.mean(axis=3)
    else:
        # 简单抽取
        idx = np.arange(0, n_samples, factor, dtype=int)
        if idx.shape[0] == 0:
            idx = np.array([0])
        
        if axis == 0:
            return data[idx]
        elif axis == 1:
            return data[:, idx]
        elif axis == 2:
            return data[:, :, idx]
    
    return data
def preprocess_eeg_data_with_downsampling(
    eeg_data, target_channels=122, target_timepoints=None,
    original_fs=500, target_fs=256, 
    downsample_method='decimate', debug=False
):
    """
    预处理EEG数据到标准形状，支持下采样
    简化的版本，不使用抗混叠滤波
    """
    if hasattr(eeg_data, 'numpy'):
        eeg_data = eeg_data.numpy()
    
    original_n_channels, original_n_times = eeg_data.shape
    
    if debug:
        print(f"  [preprocess_eeg_data_with_downsampling]")
        print(f"    原始形状: {eeg_data.shape}")
        print(f"    原始采样率: {original_fs}Hz")
        print(f"    目标采样率: {target_fs}Hz")
        print(f"    下采样因子: {original_fs/target_fs:.2f}")
        print(f"  subject_ids: {subject_ids}")
    processed_data = eeg_data.copy()
    
    # 1. 下采样（不使用抗混叠滤波）
    if target_fs < original_fs:
        downsample_factor = original_fs / target_fs
        
        if debug:
            print(f"    执行下采样: {original_fs}Hz -> {target_fs}Hz")
            print(f"    下采样因子: {downsample_factor:.2f}")
            print(f"    原始时间点: {original_n_times}")
        
        if HAS_SCIPY and downsample_method == 'decimate':
            # 使用scipy的decimate（自带滤波）
            try:
                if debug:
                    print(f"    使用scipy.decimate")
                processed_data = signal.decimate(
                    processed_data, 
                    q=int(downsample_factor), 
                    axis=1, 
                    ftype='fir',
                    zero_phase=True
                )
            except Exception as e:
                if debug:
                    print(f"    scipy.decimate失败: {e}")
                # 回退到简单抽取
                n_times = processed_data.shape[1]
                idx = np.arange(0, n_times, int(downsample_factor), dtype=int)
                if idx.shape[0] == 0:
                    idx = np.array([0])
                processed_data = processed_data[:, idx]
        else:
            # 简单抽取
            n_times = processed_data.shape[1]
            idx = np.arange(0, n_times, int(downsample_factor), dtype=int)
            if idx.shape[0] == 0:
                idx = np.array([0])
            processed_data = processed_data[:, idx]
        
        if debug:
            print(f"    下采样后形状: {processed_data.shape}")
            print(f"    下采样后时间点: {processed_data.shape[1]}")
    
    # 2. 处理通道数
    n_channels, n_times = processed_data.shape
    
    if n_channels > target_channels:
        processed = processed_data[:target_channels, :]
        if debug:
            print(f"    裁剪通道: {n_channels} -> {target_channels}")
    elif n_channels < target_channels:
        pad_width = ((0, target_channels - n_channels), (0, 0))
        processed = np.pad(processed_data, pad_width, mode='constant', constant_values=0)
        if debug:
            print(f"    填充通道: {n_channels} -> {target_channels}")
    else:
        processed = processed_data
    
    # 3. 处理时间点数
    n_times = processed.shape[1]
    
    if target_timepoints is None:
        # 如果没有指定目标时间点数，使用当前时间点数
        target_timepoints = n_times
        if debug:
            print(f"    自动设置目标时间点: {target_timepoints}")
    
    if n_times > target_timepoints:
        processed = processed[:, :target_timepoints]
        if debug:
            print(f"    裁剪时间点: {n_times} -> {target_timepoints}")
    elif n_times < target_timepoints:
        if HAS_SCIPY and n_times > 0 and n_times < target_timepoints:
            # 如果下采样后时间点太少，使用上采样
            try:
                if debug:
                    print(f"    上采样时间点: {n_times} -> {target_timepoints}")
                processed = signal.resample(processed, target_timepoints, axis=1)
            except Exception as e:
                if debug:
                    print(f"    resample失败: {e}")
                pad_width = ((0, 0), (0, target_timepoints - n_times))
                processed = np.pad(processed, pad_width, mode='constant', constant_values=0)
        else:
            pad_width = ((0, 0), (0, target_timepoints - n_times))
            processed = np.pad(processed, pad_width, mode='constant', constant_values=0)
            if debug:
                print(f"    填充时间点: {n_times} -> {target_timepoints}")
    
    # 4. 数据缩放
    processed = processed * 1000000  # 转换到微伏级
    
    # 5. 转换为float32
    processed = processed.astype(np.float32)
    
    if debug:
        print(f"    最终形状: {processed.shape}")
        if processed.size > 0:
            print(f"    数据范围: [{processed.min():.2e}, {processed.max():.2e}]")
            print(f"    数据均值: {processed.mean():.2e} ± {processed.std():.2e}")
    
    return processed



# 在 eeg_processor.py 文件的末尾（所有函数定义之后）添加
def preprocess_eeg_data(eeg_data, target_channels=122, target_timepoints=None, 
                       original_fs=500, target_fs=256, 
                       downsample_method='decimate', debug=False):
    """
    preprocess_eeg_data 函数的别名
    为了向后兼容性
    """
    return preprocess_eeg_data_with_downsampling(
        eeg_data=eeg_data,
        target_channels=target_channels,
        target_timepoints=target_timepoints,
        original_fs=original_fs,
        target_fs=target_fs,
        downsample_method=downsample_method,
        debug=debug
    )
def validate_eeg_data(processed_data):
    """验证EEG数据质量"""
    # 计算每个通道的均值和标准差
    mean = np.abs(np.mean(processed_data, axis=1))
    stds = np.std(processed_data, axis=1)
    
    # 检查是否有有效数据
    if len(mean) == 0 or len(stds) == 0:
        raise ValueError("数据为空")
    
    # 检查是否有非零均值
    if np.max(mean) <= 0:
        raise ValueError("所有通道均值为0")
    
    # 检查数值范围
    if not (mean < 100000).all():
        raise ValueError(f"均值过大: 最大{mean.max()}")
    
    if np.max(stds) <= 0:
        raise ValueError("所有通道标准差为0")
    
    if not (stds < 100000).all():
        raise ValueError(f"标准差过大: 最大{stds.max()}")
    
    return True

def verify_data_shape_and_type(data, expected_shape=(122, 1651)):
    """验证数据形状和类型"""
    if data.shape != expected_shape:
        raise ValueError(f"数据形状应为{expected_shape}，实际为{data.shape}")
    
    if data.dtype != np.float32:
        raise ValueError(f"数据类型应为np.float32，实际为{data.dtype}")
    
    return True

def map_text_labels_to_numeric(all_text_labels, text_maps):
    """将文字标签映射为数字标签"""
    numeric_labels = []
    mapping_stats = Counter()
    
    for i, text_label in enumerate(all_text_labels):
        # 实际执行映射
        numeric_label = text_maps.get(text_label, -1)
        numeric_labels.append(numeric_label)
        
        if numeric_label >= 0:
            mapping_stats['成功'] += 1
        else:
            mapping_stats['失败'] += 1
    
    return numeric_labels

def create_3category_mapping():
    """3分类映射：日常生活(0) vs 社交情感(1) vs 专业服务(2)"""
    return {
        0: 0, 13: 0, 14: 0, 18: 0, 22: 0, 23: 0, 26: 0, 35: 0, 37: 0,  # 日常生活
        1: 1, 2: 1, 6: 1, 7: 1, 9: 1, 12: 1, 15: 1, 17: 1, 24: 1, 29: 1, 34: 1, 36: 1, 38: 1,  # 社交情感
        3: 2, 4: 2, 5: 2, 8: 2, 10: 2, 11: 2, 16: 2, 19: 2, 20: 2, 21: 2, 25: 2, 27: 2, 28: 2, 30: 2, 31: 2, 32: 2, 33: 2  # 专业服务
    }

def convert_to_3category_labels(numeric_labels):
    """将39类标签转换为3类标签"""
    mapping_3cat = create_3category_mapping()
    new_labels = [mapping_3cat.get(label, -1) for label in numeric_labels]
    return new_labels

'''def process_imagine_fif_data_with_label_mapping(
    data_dir, json_path, max_files=10, debug=True, 
    target_channels=122, target_timepoints=None,  # 不再硬编码
    original_fs=500,  # 添加缺失的参数
    target_fs=256,    # 添加缺失的参数
    downsample_method='decimate'
):
    """
    处理imagine任务数据并映射标签为数字
    """
   # 使用传入的参数
    ORIGINAL_FS = original_fs
    TARGET_FS = target_fs
    
    # 自动计算目标时间点数
    if target_timepoints is None:
        # 计算时长
        original_duration = 1651 / ORIGINAL_FS  # 原始时长
        target_timepoints = int(original_duration * TARGET_FS)
    
    print(f"原始采样率: {ORIGINAL_FS}Hz")
    print(f"目标采样率: {TARGET_FS}Hz")
    print(f"下采样因子: {ORIGINAL_FS/TARGET_FS:.2f}")
    print(f"目标时间点: {target_timepoints}")
    
    if target_timepoints is None and target_fs > 0:
        # 自动计算目标时间点数（假设时长3.302秒，即1651/500≈3.302秒）
        duration_seconds = 3.302
        target_timepoints = int(duration_seconds * target_fs)
        print(f"自动计算目标时间点: {target_timepoints} (时长{duration_seconds:.3f}s × {target_fs}Hz)")
    
    if debug and target_fs < original_fs:
        downsample_factor = original_fs / target_fs
        expected_timepoints = int(target_timepoints * (target_fs / original_fs))
        print(f"下采样因子: {downsample_factor:.1f}")
        print(f"期望时间点: {expected_timepoints}")
    
    # 1. 加载textmaps映射文件
    text_maps = load_text_maps(json_path)
    if not text_maps:
        return None
    
    # 2. 查找FIF文件
    fif_files = find_imagine_fif_files(data_dir)
    if not fif_files:
        print("❌ 未找到imagine任务的FIF文件")
        return None
    
    print(f"找到 {len(fif_files)} 个imagine任务FIF文件")
    
    # 限制处理文件数
    test_files = fif_files[:max_files]
    print(f"处理前 {len(test_files)} 个imagine任务文件")
    
    all_data = []  # 存储处理后的EEG数据
    all_text_labels = []  # 存储文本标签
    file_info = []  # 存储文件信息
    
    for i, file_path in enumerate(test_files):
        try:
            if debug:
                print(f"\n--- 处理文件 {i+1}/{len(test_files)}: {os.path.basename(file_path)} ---")
            
            # 读取epoch文件
            epochs = mne.read_epochs(file_path, preload=True, verbose=False)
            n_epochs = len(epochs)
            
            if debug:
                print(f"  Trials数: {n_epochs}")
            
            # 提取Word标签
            trial_labels = extract_word_labels_from_metadata(epochs, n_epochs)
            
            if debug:
                label_counts = Counter(trial_labels)
                print(f"  Word标签分布: 共{len(label_counts)}种，示例: {dict(list(label_counts.items())[:5])}")
            
            # 获取EEG数据
            picks = mne.pick_types(epochs.info, eeg=True, exclude='bads')
            if len(picks) == 0:
                if debug:
                    print("  ⚠ 跳过: 没有EEG通道")
                continue
            
            eeg_data = epochs.get_data(picks=picks)
            
            if debug:
                print(f"  原始数据形状: {eeg_data.shape}")
            
            # 处理每个trial
            trial_count = 0
            successful_trials = 0
            
            for j in range(n_epochs):
                try:
                    trial_data = eeg_data[j]  # (n_channels, n_times)
                    word_label = trial_labels[j]
                    
                    # 1. 预处理到标准形状
                    processed_data = preprocess_eeg_data_with_downsampling(
                        trial_data, 
                        target_channels=target_channels, 
                        target_timepoints=target_timepoints,
                        original_fs=original_fs,
                        target_fs=target_fs,
                        downsample_method=downsample_method,
                        debug=debug and j < 1  # 只调试第一个样本
                    )
                    
                    # 2. 验证数据质量
                    validate_eeg_data(processed_data)
                    
                    # 3. 转换为torch.Tensor
                    tensor_data = torch.tensor(processed_data)
                    
                    all_data.append(tensor_data)
                    all_text_labels.append(word_label)
                    trial_count += 1
                    successful_trials += 1
                    
                except Exception as e:
                    if debug and j < 3:  # 只显示前几个错误
                        print(f"    ❌ Trial {j} 处理失败: {e}")
                    trial_count += 1
                    continue
            
            file_info.append({
                'file': file_path,
                'n_epochs': n_epochs,
                'processed_trials': trial_count,
                'successful_trials': successful_trials
            })
            
            if debug:
                print(f"  ✓ 成功处理 {successful_trials}/{trial_count} 个trials")
            
        except Exception as e:
            if debug:
                print(f"  ❌ 文件处理失败: {e}")
            continue
    
    if not all_data:
        print("❌ 所有trials处理失败")
        return None
    
    print(f"\n总共处理了 {len(all_data)} 个有效的trials")
    
    # 3. 映射文字标签为数字标签
    numeric_labels = map_text_labels_to_numeric(all_text_labels, text_maps)
    
    # 4. 过滤有效样本（映射成功的）
    valid_indices = [i for i, label in enumerate(numeric_labels) if label >= 0]
    
    if not valid_indices:
        print("❌ 无有效映射标签的样本")
        return None
    
    valid_data = [all_data[i] for i in valid_indices]
    valid_numeric_labels = [numeric_labels[i] for i in valid_indices]
    valid_text_labels = [all_text_labels[i] for i in valid_indices]
    
    print(f"\n过滤后有效样本: {len(valid_data)} 个")
    print(f"原始样本数: {len(all_data)}")
    print(f"过滤比例: {len(valid_data)/len(all_data)*100:.1f}%")
    
    # 5. 统计信息
    label_counts = Counter(valid_numeric_labels)
    print("\n39分类分布:")
    for label_id in sorted(label_counts.keys()):
        count = label_counts[label_id]
        percentage = count / len(valid_numeric_labels) * 100
        print(f"  类别{label_id}: {count} 样本 ({percentage:.1f}%)")
    
    # 6. 转换为3分类
    three_category_labels = convert_to_3category_labels(valid_numeric_labels)
    
    # 移除映射失败的样本（应该不会发生，但安全起见）
    valid_indices_3cat = [i for i, label in enumerate(three_category_labels) if label >= 0]
    final_data = [valid_data[i] for i in valid_indices_3cat]
    final_numeric_labels = [three_category_labels[i] for i in valid_indices_3cat]
    final_text_labels = [valid_text_labels[i] for i in valid_indices_3cat]
    
    print(f"\n3分类分布:")
    label_counts_3cat = Counter(final_numeric_labels)
    for label_id in sorted(label_counts_3cat.keys()):
        count = label_counts_3cat[label_id]
        percentage = count / len(final_numeric_labels) * 100
        label_names = {0: "日常生活", 1: "社交情感", 2: "专业服务"}
        print(f"  {label_names[label_id]}({label_id}): {count} 样本 ({percentage:.1f}%)")
    
    # 7. 转换为Tensor
    input_tensors = torch.stack(final_data, dim=0)  # (batch_size, target_channels, actual_timepoints)
    
    # 获取实际的时间点数
    actual_timepoints = input_tensors.shape[2]
    
    label_tensors = torch.tensor(final_numeric_labels, dtype=torch.long)  # (batch_size,)
    
    # 创建注意力掩码（全1，因为EEG是固定长度）
    attention_masks = torch.ones(len(final_data), actual_timepoints, dtype=torch.long)
    
    # 8. 返回数据集字典
    dataset = {
        'input_features': input_tensors,  # (batch_size, target_channels, actual_timepoints)
        'numeric_labels': label_tensors,  # (batch_size,)
        'text_labels': final_text_labels,  # list of str
        'attention_mask': attention_masks,  # (batch_size, actual_timepoints)
        'num_classes': 3,  # 3分类
        'feature_dim': target_channels,  # 特征维度
        'seq_len': actual_timepoints,  # 实际序列长度
        'sample_count': len(final_data),
        'downsample_factor': original_fs / target_fs if target_fs < original_fs else 1.0,
        'original_fs': original_fs,
        'target_fs': target_fs
    }
    
    print(f"\n✓ 数据集创建完成")
    print(f"  输入特征形状: {input_tensors.shape}")
    print(f"  标签形状: {label_tensors.shape}")
    print(f"  类别数量: 3")
    print(f"  实际序列长度: {actual_timepoints}")
    print(f"  特征维度: {target_channels}")
    if target_fs < original_fs:
        print(f"  下采样因子: {dataset['downsample_factor']:.1f}")
    print(f"  采样率: {original_fs}Hz -> {target_fs}Hz")
    
    return dataset'''
'''12.7:18:00   --------------------------------------------------这版是读取所有被试--------------------------------
def process_imagine_fif_data_with_label_mapping(
    data_dir, json_path, max_files=10, debug=True, 
    target_channels=122, target_timepoints=None,
    original_fs=500,
    target_fs=256,
    downsample_method='decimate',
    subject_ids=None  # 新增：支持多个被试
):
    """
    处理imagine任务数据并映射标签为数字
    """
    # 使用传入的参数
    ORIGINAL_FS = original_fs
    TARGET_FS = target_fs
    
    # 自动计算目标时间点数
    if target_timepoints is None:
        # 计算时长
        original_duration = 1651 / ORIGINAL_FS
        target_timepoints = int(original_duration * TARGET_FS)
    
    print(f"\n=== EEG数据处理配置 ===")
    print(f"数据目录: {data_dir}")
    print(f"原始采样率: {ORIGINAL_FS}Hz")
    print(f"目标采样率: {TARGET_FS}Hz")
    print(f"下采样因子: {ORIGINAL_FS/TARGET_FS:.2f}")
    print(f"目标时间点: {target_timepoints}")
    print(f"目标通道数: {target_channels}")
    
    # 处理subject_ids参数
    if subject_ids is None:
        # 如果没有指定subject_ids，查找所有被试
        subject_ids = find_all_subjects(data_dir)
        if not subject_ids:
            print(f"❌ 错误: 在 {data_dir} 中未找到被试目录")
            return None
    elif isinstance(subject_ids, str):
        # 如果是字符串，按逗号分割
        subject_ids = [s.strip() for s in subject_ids.split(',')]
    elif not isinstance(subject_ids, list):
        # 如果是单个值，转换为列表
        subject_ids = [str(subject_ids)]
    
    print(f"处理被试: {subject_ids}")
    print(f"被试数量: {len(subject_ids)}")
    
    # 1. 加载textmaps映射文件
    text_maps = load_text_maps(json_path)
    if not text_maps:
        return None
    
    all_data = []  # 存储处理后的EEG数据
    all_text_labels = []  # 存储文本标签
    all_subject_ids = []  # 存储被试ID
    
    total_files_processed = 0
    total_trials_processed = 0
    
    # 遍历所有被试
    for subject_idx, subject_id in enumerate(subject_ids):
        subject_path = os.path.join(data_dir, subject_id)
        
        if not os.path.exists(subject_path):
            print(f"⚠ 警告: 被试 {subject_id} 的路径不存在: {subject_path}")
            continue
        
        print(f"\n--- 处理被试 {subject_idx+1}/{len(subject_ids)}: {subject_id} ---")
        print(f"路径: {subject_path}")
        
        # 查找该被试的FIF文件
        fif_files = find_imagine_fif_files(subject_path)
        if not fif_files:
            print(f"⚠ 被试 {subject_id}: 未找到imagine任务的FIF文件")
            continue
        
        print(f"被试 {subject_id}: 找到 {len(fif_files)} 个imagine任务FIF文件")
        
        # 限制处理文件数
        test_files = fif_files[:max_files]
        print(f"被试 {subject_id}: 处理前 {len(test_files)} 个文件")
        
        subject_trial_count = 0
        subject_successful_trials = 0
        
        for i, file_path in enumerate(test_files):
            try:
                if debug and i < 3:  # 只显示前几个文件的详细信息
                    print(f"  处理文件 {i+1}/{len(test_files)}: {os.path.basename(file_path)}")
                
                # 读取epoch文件
                epochs = mne.read_epochs(file_path, preload=True, verbose=False)
                n_epochs = len(epochs)
                
                if debug and i < 3:
                    print(f"    包含 {n_epochs} 个trials")
                
                # 提取Word标签
                trial_labels = extract_word_labels_from_metadata(epochs, n_epochs)
                
                if debug and i < 3 and trial_labels:
                    label_counts = Counter(trial_labels)
                    print(f"    前3个trials的Word标签: {trial_labels[:3]}")
                
                # 获取EEG数据
                picks = mne.pick_types(epochs.info, eeg=True, exclude='bads')
                if len(picks) == 0:
                    if debug and i < 3:
                        print("    ⚠ 跳过: 没有EEG通道")
                    continue
                
                eeg_data = epochs.get_data(picks=picks)
                
                if debug and i < 3:
                    print(f"    原始数据形状: {eeg_data.shape}")
                
                # 处理每个trial
                for j in range(n_epochs):
                    try:
                        trial_data = eeg_data[j]  # (n_channels, n_times)
                        word_label = trial_labels[j] if j < len(trial_labels) else "unknown"
                        
                        # 预处理到标准形状
                        processed_data = preprocess_eeg_data_with_downsampling(
                            trial_data, 
                            target_channels=target_channels, 
                            target_timepoints=target_timepoints,
                            original_fs=original_fs,
                            target_fs=target_fs,
                            downsample_method=downsample_method,
                            debug=debug and i < 1 and j < 1  # 只调试第一个样本
                        )
                        
                        # 验证数据质量
                        validate_eeg_data(processed_data)
                        
                        # 转换为torch.Tensor
                        tensor_data = torch.tensor(processed_data)
                        
                        all_data.append(tensor_data)
                        all_text_labels.append(word_label)
                        all_subject_ids.append(subject_id)
                        subject_trial_count += 1
                        subject_successful_trials += 1
                        total_trials_processed += 1
                        
                    except Exception as e:
                        if debug and i < 1 and j < 3:  # 只显示前几个错误
                            print(f"    ❌ Trial {j} 处理失败: {e}")
                        subject_trial_count += 1
                        continue
                
                if debug and i < 3:
                    print(f"    成功处理 {subject_successful_trials}/{subject_trial_count} 个trials")
                
                total_files_processed += 1
                
            except Exception as e:
                if debug and i < 3:
                    print(f"  ❌ 文件处理失败: {e}")
                continue
        
        print(f"✓ 被试 {subject_id} 处理完成:")
        print(f"  总trials: {subject_trial_count}")
        print(f"  成功trials: {subject_successful_trials}")
        if subject_trial_count > 0:
            print(f"  成功率: {subject_successful_trials/subject_trial_count*100:.1f}%")
    
    if not all_data:
        print("❌ 所有trials处理失败")
        return None
    
    print(f"\n=== 总体处理结果 ===")
    print(f"处理的被试数: {len(subject_ids)}")
    print(f"处理的文件数: {total_files_processed}")
    print(f"总trials数: {total_trials_processed}")
    print(f"有效样本数: {len(all_data)}")
    
    # 统计被试分布
    subject_counts = Counter(all_subject_ids)
    print("\n被试样本分布:")
    for subj_id, count in sorted(subject_counts.items()):
        percentage = count/len(all_data)*100 if len(all_data) > 0 else 0
        print(f"  {subj_id}: {count} 样本 ({percentage:.1f}%)")
    
    # 2. 映射文字标签为数字标签
    numeric_labels = map_text_labels_to_numeric(all_text_labels, text_maps)
    
    # 3. 过滤有效样本（映射成功的）
    valid_indices = [i for i, label in enumerate(numeric_labels) if label >= 0]
    
    if not valid_indices:
        print("❌ 无有效映射标签的样本")
        return None
    
    valid_data = [all_data[i] for i in valid_indices]
    valid_numeric_labels = [numeric_labels[i] for i in valid_indices]
    valid_text_labels = [all_text_labels[i] for i in valid_indices]
    valid_subject_ids = [all_subject_ids[i] for i in valid_indices]
    
    print(f"\n过滤后有效样本: {len(valid_data)} 个")
    print(f"原始样本数: {len(all_data)}")
    if len(all_data) > 0:
        print(f"过滤比例: {len(valid_data)/len(all_data)*100:.1f}%")
    
    # 4. 转换为3分类
    three_category_labels = convert_to_3category_labels(valid_numeric_labels)
    
    # 移除映射失败的样本
    valid_indices_3cat = [i for i, label in enumerate(three_category_labels) if label >= 0]
    final_data = [valid_data[i] for i in valid_indices_3cat]
    final_numeric_labels = [three_category_labels[i] for i in valid_indices_3cat]
    final_text_labels = [valid_text_labels[i] for i in valid_indices_3cat]
    final_subject_ids = [valid_subject_ids[i] for i in valid_indices_3cat]
    
    print(f"\n3分类分布:")
    label_counts_3cat = Counter(final_numeric_labels)
    label_names = {0: "日常生活", 1: "社交情感", 2: "专业服务"}
    for label_id in sorted(label_counts_3cat.keys()):
        count = label_counts_3cat[label_id]
        percentage = count/len(final_numeric_labels)*100 if len(final_numeric_labels) > 0 else 0
        print(f"  {label_names[label_id]}({label_id}): {count} 样本 ({percentage:.1f}%)")
    
    # 5. 转换为Tensor
    input_tensors = torch.stack(final_data, dim=0)  # (batch_size, target_channels, actual_timepoints)
    
    # 获取实际的时间点数
    actual_timepoints = input_tensors.shape[2]
    
    label_tensors = torch.tensor(final_numeric_labels, dtype=torch.long)  # (batch_size,)
    
    # 创建注意力掩码
    attention_masks = torch.ones(len(final_data), actual_timepoints, dtype=torch.long)
    
    # 6. 返回数据集字典
    dataset = {
        'input_features': input_tensors,  # (batch_size, target_channels, actual_timepoints)
        'numeric_labels': label_tensors,  # (batch_size,)
        'text_labels': final_text_labels,  # list of str
        'subject_ids': final_subject_ids,  # list of str
        'attention_mask': attention_masks,  # (batch_size, actual_timepoints)
        'num_classes': 3,  # 3分类
        'feature_dim': target_channels,  # 特征维度
        'seq_len': actual_timepoints,  # 实际序列长度
        'sample_count': len(final_data),
        'subject_count': len(set(final_subject_ids)),
        'downsample_factor': original_fs / target_fs if target_fs < original_fs else 1.0,
        'original_fs': original_fs,
        'target_fs': target_fs
    }
    
    print(f"\n✓ 数据集创建完成")
    print(f"  输入特征形状: {input_tensors.shape}")
    print(f"  标签形状: {label_tensors.shape}")
    print(f"  被试数: {dataset['subject_count']}")
    print(f"  样本数: {len(final_data)}")
    print(f"  类别数量: 3")
    print(f"  实际序列长度: {actual_timepoints}")
    print(f"  特征维度: {target_channels}")
    if target_fs < original_fs:
        print(f"  下采样因子: {dataset['downsample_factor']:.1f}")
    print(f"  采样率: {original_fs}Hz -> {target_fs}Hz")
    
    return dataset'''
def process_imagine_fif_data_with_label_mapping(
    data_dir, json_path, max_files=10, debug=True, 
    target_channels=122, target_timepoints=None,
    original_fs=500, target_fs=256, 
    downsample_method='decimate',
    subject_ids=None  # 只保留这个参数
):
    """
    处理imagine任务数据并映射标签为数字
    
    Args:
        data_dir: 数据目录
        json_path: 文本映射JSON文件路径
        max_files: 每个被试最大处理文件数
        debug: 是否输出调试信息
        target_channels: 目标通道数
        target_timepoints: 目标时间点数
        original_fs: 原始采样率
        target_fs: 目标采样率
        downsample_method: 下采样方法
        subject_ids: 被试ID列表
    """
    # 使用传入的参数
    ORIGINAL_FS = original_fs
    TARGET_FS = target_fs
    
    # 自动计算目标时间点数
    if target_timepoints is None:
        # 计算时长
        original_duration = 1651 / ORIGINAL_FS
        target_timepoints = int(original_duration * TARGET_FS)
    
    print(f"\n=== EEG数据处理配置 ===")
    print(f"数据目录: {data_dir}")
    print(f"JSON映射: {json_path}")
    print(f"原始采样率: {ORIGINAL_FS}Hz")
    print(f"目标采样率: {TARGET_FS}Hz")
    print(f"目标时间点: {target_timepoints}")
    print(f"目标通道数: {target_channels}")
    print(f"最大文件数: {max_files}")
    
    # 1. 加载textmaps映射文件
    text_maps = load_text_maps(json_path)
    if not text_maps:
        return None
    
    # 2. 处理subject_ids参数
    if subject_ids is None:
        # 如果未指定subject_ids，查找所有被试
        all_subject_dirs = find_all_subjects(data_dir)
        if not all_subject_dirs:
            print(f"❌ 错误: 在 {data_dir} 中未找到被试目录")
            return None
        subjects_to_process = all_subject_dirs
        print(f"使用全部被试: 共{len(subjects_to_process)}个")
    else:
        # 转换subject_ids为列表
        if isinstance(subject_ids, str):
            # 如果是字符串，按逗号分割
            subjects_to_process = [s.strip() for s in subject_ids.split(',')]
        elif isinstance(subject_ids, list):
            subjects_to_process = subject_ids
        elif isinstance(subject_ids, int):
            # 如果是单个整数，转换为sub-xxx格式
            subjects_to_process = [f"sub-{subject_ids:03d}"]
        else:
            print(f"❌ 错误: subject_ids参数类型错误: {type(subject_ids)}")
            return None
        print(f"指定处理被试: {subjects_to_process}")
    
    if not subjects_to_process:
        print("❌ 错误: 没有要处理的被试")
        return None
    
    # 检查被试目录是否存在
    valid_subjects = []
    for subj_id in subjects_to_process:
        subj_path = os.path.join(data_dir, subj_id)
        if os.path.exists(subj_path):
            valid_subjects.append(subj_id)
        else:
            print(f"⚠ 警告: 被试目录不存在: {subj_path}")
    
    subjects_to_process = valid_subjects
    if not subjects_to_process:
        print("❌ 错误: 没有有效的被试目录")
        return None
    
    print(f"将处理 {len(subjects_to_process)} 个被试:")
    for i, subj_id in enumerate(subjects_to_process[:20]):
        print(f"  {i+1:3d}. {subj_id}")
    if len(subjects_to_process) > 20:
        print(f"  ... 和 {len(subjects_to_process)-20} 个更多被试")
    
    all_data = []  # 存储处理后的EEG数据
    all_text_labels = []  # 存储文本标签
    all_subject_ids = []  # 存储被试ID
    
    total_files_processed = 0
    total_trials_processed = 0
    total_successful_trials = 0
    
    # 3. 遍历所有被试
    for subject_idx, subject_id in enumerate(subjects_to_process):
        subject_path = os.path.join(data_dir, subject_id)
        
        if debug:
            print(f"\n--- 处理被试 {subject_idx+1}/{len(subjects_to_process)}: {subject_id} ---")
            print(f"路径: {subject_path}")
        else:
            if (subject_idx + 1) % 10 == 0 or subject_idx == 0 or subject_idx == len(subjects_to_process) - 1:
                print(f"处理被试 {subject_idx+1}/{len(subjects_to_process)}: {subject_id}")
        
        # 查找该被试的FIF文件
        fif_files = find_imagine_fif_files(subject_path)
        if not fif_files:
            if debug:
                print(f"⚠ 被试 {subject_id}: 未找到imagine任务的FIF文件")
            continue
        
        if debug:
            print(f"被试 {subject_id}: 找到 {len(fif_files)} 个imagine任务FIF文件")
        
        # 限制处理文件数
        if max_files > 0 and len(fif_files) > max_files:
            files_to_process = fif_files[:max_files]
            if debug:
                print(f"被试 {subject_id}: 限制处理前 {len(files_to_process)} 个文件 (共{len(fif_files)}个)")
        else:
            files_to_process = fif_files
        
        subject_trial_count = 0
        subject_successful_trials = 0
        
        for i, file_path in enumerate(files_to_process):
            try:
                if debug and i < 3:  # 只显示前几个文件的详细信息
                    print(f"  处理文件 {i+1}/{len(files_to_process)}: {os.path.basename(file_path)}")
                
                # 读取epoch文件
                epochs = mne.read_epochs(file_path, preload=True, verbose=False)
                n_epochs = len(epochs)
                
                if debug and i < 3:
                    print(f"    包含 {n_epochs} 个trials")
                
                # 提取Word标签
                trial_labels = extract_word_labels_from_metadata(epochs, n_epochs)
                
                if debug and i < 3 and trial_labels:
                    label_counts = Counter(trial_labels)
                    print(f"    前3个trials的Word标签: {trial_labels[:3]}")
                
                # 获取EEG数据
                picks = mne.pick_types(epochs.info, eeg=True, exclude='bads')
                if len(picks) == 0:
                    if debug and i < 3:
                        print("    ⚠ 跳过: 没有EEG通道")
                    continue
                
                eeg_data = epochs.get_data(picks=picks)
                
                if debug and i < 3:
                    print(f"    原始数据形状: {eeg_data.shape}")
                
                # 处理每个trial
                for j in range(n_epochs):
                    try:
                        trial_data = eeg_data[j]  # (n_channels, n_times)
                        word_label = trial_labels[j] if j < len(trial_labels) else "unknown"
                        
                        # 预处理到标准形状
                        processed_data = preprocess_eeg_data_with_downsampling(
                            trial_data, 
                            target_channels=target_channels, 
                            target_timepoints=target_timepoints,
                            original_fs=original_fs,
                            target_fs=target_fs,
                            downsample_method=downsample_method,
                            debug=debug and i < 1 and j < 1  # 只调试第一个样本
                        )
                        
                        # 验证数据质量
                        validate_eeg_data(processed_data)
                        
                        # 转换为torch.Tensor
                        tensor_data = torch.tensor(processed_data)
                        
                        all_data.append(tensor_data)
                        all_text_labels.append(word_label)
                        all_subject_ids.append(subject_id)
                        subject_trial_count += 1
                        subject_successful_trials += 1
                        total_trials_processed += 1
                        total_successful_trials += 1
                        
                    except Exception as e:
                        if debug and i < 1 and j < 3:  # 只显示前几个错误
                            print(f"    ❌ Trial {j} 处理失败: {e}")
                        subject_trial_count += 1
                        continue
                
                if debug and i < 3:
                    print(f"    成功处理 {subject_successful_trials}/{subject_trial_count} 个trials")
                
                total_files_processed += 1
                
            except Exception as e:
                if debug and i < 3:
                    print(f"  ❌ 文件处理失败: {e}")
                continue
        
        if debug and subject_trial_count > 0:
            print(f"✓ 被试 {subject_id} 处理完成:")
            print(f"  总trials: {subject_trial_count}")
            print(f"  成功trials: {subject_successful_trials}")
            if subject_trial_count > 0:
                print(f"  成功率: {subject_successful_trials/subject_trial_count*100:.1f}%")
    
    if not all_data:
        print("❌ 所有trials处理失败")
        return None
    
    print(f"\n=== 总体处理结果 ===")
    print(f"处理的被试数: {len(subjects_to_process)}")
    print(f"实际处理被试数: {len(set(all_subject_ids))}")
    print(f"处理的文件数: {total_files_processed}")
    print(f"总trials数: {total_trials_processed}")
    print(f"有效样本数: {len(all_data)}")
    if total_trials_processed > 0:
        print(f"总体成功率: {total_successful_trials/total_trials_processed*100:.1f}%")
    
    # 统计被试分布
    if all_subject_ids:
        subject_counts = Counter(all_subject_ids)
        print(f"\n被试样本分布:")
        for subj_id, count in sorted(subject_counts.items()):
            percentage = count/len(all_data)*100 if len(all_data) > 0 else 0
            print(f"  {subj_id}: {count} 样本 ({percentage:.1f}%)")
    
    # 4. 映射文字标签为数字标签
    numeric_labels = map_text_labels_to_numeric(all_text_labels, text_maps)
    
    # 5. 过滤有效样本（映射成功的）
    valid_indices = [i for i, label in enumerate(numeric_labels) if label >= 0]
    
    if not valid_indices:
        print("❌ 无有效映射标签的样本")
        return None
    
    valid_data = [all_data[i] for i in valid_indices]
    valid_numeric_labels = [numeric_labels[i] for i in valid_indices]
    valid_text_labels = [all_text_labels[i] for i in valid_indices]
    valid_subject_ids = [all_subject_ids[i] for i in valid_indices]
    
    print(f"\n过滤后有效样本: {len(valid_data)} 个")
    print(f"原始样本数: {len(all_data)}")
    if len(all_data) > 0:
        print(f"标签映射成功率: {len(valid_data)/len(all_data)*100:.1f}%")
    
    # 6. 转换为3分类
    three_category_labels = convert_to_3category_labels(valid_numeric_labels)
    
    # 移除映射失败的样本
    valid_indices_3cat = [i for i, label in enumerate(three_category_labels) if label >= 0]
    final_data = [valid_data[i] for i in valid_indices_3cat]
    final_numeric_labels = [three_category_labels[i] for i in valid_indices_3cat]
    final_text_labels = [valid_text_labels[i] for i in valid_indices_3cat]
    final_subject_ids = [valid_subject_ids[i] for i in valid_indices_3cat]
    
    if not final_data:
        print("❌ 转换为3分类后无有效样本")
        return None
    
    print(f"\n3分类分布:")
    label_counts_3cat = Counter(final_numeric_labels)
    label_names = {0: "日常生活", 1: "社交情感", 2: "专业服务"}
    for label_id in sorted(label_counts_3cat.keys()):
        count = label_counts_3cat[label_id]
        percentage = count/len(final_numeric_labels)*100 if len(final_numeric_labels) > 0 else 0
        print(f"  {label_names[label_id]}({label_id}): {count} 样本 ({percentage:.1f}%)")
    
    # 7. 转换为Tensor
    input_tensors = torch.stack(final_data, dim=0)  # (batch_size, target_channels, actual_timepoints)
    
    # 获取实际的时间点数
    actual_timepoints = input_tensors.shape[2]
    
    label_tensors = torch.tensor(final_numeric_labels, dtype=torch.long)  # (batch_size,)
    
    # 创建注意力掩码
    attention_masks = torch.ones(len(final_data), actual_timepoints, dtype=torch.long)
    
    # 8. 返回数据集字典
    dataset = {
        'input_features': input_tensors,  # (batch_size, target_channels, actual_timepoints)
        'numeric_labels': label_tensors,  # (batch_size,)
        'text_labels': final_text_labels,  # list of str
        'subject_ids': final_subject_ids,  # list of str
        'attention_mask': attention_masks,  # (batch_size, actual_timepoints)
        'num_classes': 3,  # 3分类
        'feature_dim': target_channels,  # 特征维度
        'seq_len': actual_timepoints,  # 实际序列长度
        'sample_count': len(final_data),
        'subject_count': len(set(final_subject_ids)),
        'downsample_factor': original_fs / target_fs if target_fs < original_fs else 1.0,
        'original_fs': original_fs,
        'target_fs': target_fs,
        'processed_subjects': sorted(set(final_subject_ids))  # 记录处理的被试
    }
    
    print(f"\n✓ 数据集创建完成")
    print(f"  输入特征形状: {input_tensors.shape}")
    print(f"  标签形状: {label_tensors.shape}")
    print(f"  被试数: {dataset['subject_count']}")
    print(f"  样本数: {len(final_data)}")
    print(f"  类别数量: 3")
    print(f"  实际序列长度: {actual_timepoints}")
    print(f"  特征维度: {target_channels}")
    if target_fs < original_fs:
        print(f"  下采样因子: {dataset['downsample_factor']:.1f}")
    print(f"  采样率: {original_fs}Hz -> {target_fs}Hz")
    print(f"  处理的被试: {', '.join(dataset['processed_subjects'])}")
    
    return dataset

def find_all_subjects(data_dir):
    """查找数据目录下的所有被试文件夹"""
    subjects = []
    try:
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path) and item.startswith('sub-'):
                subjects.append(item)
    except Exception as e:
        print(f"查找被试文件夹失败: {e}")
    
    subjects.sort()
    return subjects