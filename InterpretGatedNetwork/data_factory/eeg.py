# data_provider/eeg.py
import os
import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import json
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# 在 eeg.py 的顶部修改导入
import os
import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import json
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# 修改这部分导入代码
try:
    from .eeg_processor import (
        process_imagine_fif_data_with_label_mapping,
        load_text_maps,
        find_imagine_fif_files,
        validate_eeg_data,
        verify_data_shape_and_type,
        map_text_labels_to_numeric,
        create_3category_mapping,
        convert_to_3category_labels
    )
    print("✓ 成功导入 eeg_processor 模块")
    
    # 检查是否有 preprocess_eeg_data_with_downsampling
    try:
        from .eeg_processor import preprocess_eeg_data_with_downsampling
        # 创建别名
        preprocess_eeg_data = preprocess_eeg_data_with_downsampling
        print("✓ 成功导入 preprocess_eeg_data_with_downsampling 并创建别名")
    except ImportError:
        print("⚠ 注意: eeg_processor中没有preprocess_eeg_data_with_downsampling函数")
        preprocess_eeg_data = None
        
except ImportError as e:
    print(f"⚠ 警告: 导入 eeg_processor 模块失败: {e}")
    print("⚠ 将使用本地函数定义")
    
    # 在本地定义函数
    def process_imagine_fif_data_with_label_mapping(*args, **kwargs):
        raise ImportError("eeg_processor 模块未找到")
    
    def convert_to_3category_labels(numeric_labels):
        """将39类标签转换为3类标签"""
        mapping_3cat = create_3category_mapping()
        new_labels = [mapping_3cat.get(label, -1) for label in numeric_labels]
        return new_labels
    
    def create_3category_mapping():
        """3分类映射：日常生活(0) vs 社交情感(1) vs 专业服务(2)"""
        return {
            0: 0, 13: 0, 14: 0, 18: 0, 22: 0, 23: 0, 26: 0, 35: 0, 37: 0,  # 日常生活
            1: 1, 2: 1, 6: 1, 7: 1, 9: 1, 12: 1, 15: 1, 17: 1, 24: 1, 29: 1, 34: 1, 36: 1, 38: 1,  # 社交情感
            3: 2, 4: 2, 5: 2, 8: 2, 10: 2, 11: 2, 16: 2, 19: 2, 20: 2, 21: 2, 25: 2, 27: 2, 28: 2, 30: 2, 31: 2, 32: 2, 33: 2  # 专业服务
        }

# 导入现有的数据处理工具
from .uea import Normalizer, interpolate_missing, subsample


def eeg_collate_fn(data, max_len=None):
    """
    EEG专用collate函数
    支持动态长度
    """
    batch_size = len(data)
    features, labels = zip(*data)
    
    # 获取序列长度
    seq_len = features[0].shape[0]  # 转置后是(seq_len, feat_dim)
    
    # 直接stack，因为EEG是固定长度
    X = torch.stack(features, dim=0)  # (batch_size, seq_len, feat_dim)
    targets = torch.stack(labels, dim=0)  # (batch_size,)
    
    # 创建全1的padding_mask
    padding_masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    return X, targets, padding_masks

class EEGDataset(Dataset):
    """
    EEG想象任务数据集，39分类版本
    适配现有框架的数据集接口
    """
    def __init__(self, root_path, flag='train', size=None, features='S', 
                 data_path='', target='OT', scale=True, timeenc=0, freq='h', 
                 seasonal_patterns=None, nbins=10, bin_edges=None, 
                 json_path=None, max_files=10, debug=False, 
                 test_size=0.2, val_size=0.1, random_seed=42,args=None):
        
        """
        Args:
            root_path: 数据根目录
            flag: 'train', 'val', 'test'
            size: [seq_len, label_len, pred_len] (分类任务用不到label_len和pred_len)
            json_path: textmaps.json路径
            max_files: 最大处理文件数
            test_size: 测试集比例
            val_size: 验证集比例
        """
        try:
            # 处理flag大小写
            if isinstance(flag, str):
                flag = flag.lower()
                if flag == 'validation':
                    flag = 'val'

            if flag not in ['train', 'val', 'test']:
                if debug:
                    print(f"⚠ 警告: flag={flag} 不是标准值，使用默认值 'train'")
                flag = 'train'

            # 保存参数
            self.flag = flag
            self.seq_len = None  # 不再硬编码
            self.label_len = size[1] if size else 0
            self.pred_len = size[2] if size else 0
            self.scale = scale
            self.json_path = json_path
            self.max_files = max_files
            self.debug = debug
            self.test_size = test_size
            self.val_size = val_size
            self.random_seed = random_seed
            
            # === 固定参数 ===
            self.original_fs = 500  # 固定: 原始采样率500Hz
            self.target_fs = 256    # 固定: 目标采样率256Hz
            self.target_channels = 122  # 固定: 目标通道数122
            self.downsample_method = 'decimate'  # 固定: 下采样方法
            self.target_timepoints = None  # 稍后从数据计算

            # 设置随机种子
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

            if debug:
                print(f"\n{'='*60}")
                print("EEGDataset初始化")
                print(f"参数:")
                print(f"  flag: {flag}")
                print(f"  root_path: {root_path}")
                print(f"  target_fs: {self.target_fs}Hz")
                print(f"  json_path: {json_path}")
                print(f"  original_fs: {self.original_fs}Hz (固定)")
                print(f"  target_fs: {self.target_fs}Hz (固定)")
                print(f"  target_channels: {self.target_channels}")

            # 验证必要参数
            if json_path is None or not os.path.exists(json_path):
                raise FileNotFoundError(f"textmaps.json文件不存在: {json_path}")

            # 加载和处理EEG数据
            print("正在加载EEG数据...")
            self.data_dict = self._load_eeg_data(root_path)
            if not self.data_dict:
                raise ValueError("无法加载EEG数据")
            self.samples = self._prepare_samples()
            print(f"有效样本数量: {len(self.samples)}")
            # 从data_dict获取实际尺寸
            if 'input_features' in self.data_dict and len(self.data_dict['input_features']) > 0:
                # 获取实际形状
                n_samples, n_channels, n_times = self.data_dict['input_features'].shape
                self.target_timepoints = n_times
                self.target_channels = n_channels

                # 更新seq_len
                if size and size[0] is not None:
                    self.seq_len = size[0]
                else:
                    self.seq_len = n_times

                # 添加max_seq_len属性
                self.max_seq_len = n_times

                if self.debug:
                    print(f"实际数据形状: {self.data_dict['input_features'].shape}")
                    print(f"seq_len设置为: {self.seq_len}")
                    print(f"max_seq_len: {self.max_seq_len}")
            else:
                # 使用默认值
                if size and size[0] is not None:
                    self.seq_len = size[0]
                else:
                    # 计算默认值: 1651/500 ≈ 3.302秒 × 256Hz ≈ 845
                    self.seq_len = int(1651 * self.target_fs / self.original_fs)
                self.max_seq_len = self.seq_len
                print(f"⚠ 警告: 无法从data_dict获取尺寸，使用默认值: {self.seq_len}")

            if self.debug:
                print(f"✓ 设置max_seq_len: {self.max_seq_len}")
                print("✓ 数据集创建完成")
                if 'input_features' in self.data_dict:
                    print(f"  输入特征形状: {self.data_dict['input_features'].shape}")
                if 'numeric_labels' in self.data_dict:
                    print(f"  标签形状: {self.data_dict['numeric_labels'].shape}")
                if 'num_classes' in self.data_dict:
                    print(f"  类别数量: {self.data_dict['num_classes']}")

            # 数据标准化
            if self.scale:
                self._setup_normalizer()

            # 数据划分
            self.samples = self._split_samples_by_flag()

            if self.debug and len(self.samples) > 0:
                print(f"\n✓ EEGDataset ({flag}集) 初始化完成:")
                print(f"  样本数量: {len(self.samples)}")
                sample = self.samples[0]
                print(f"  特征形状: {sample['features'].shape}")
                print(f"  类别数量: {self.data_dict.get('num_classes', 'unknown')}")
                print(f"  seq_len: {self.seq_len}")
                print(f"  max_seq_len: {self.max_seq_len}")

        except Exception as e:
            print(f"\n❌ EEGDataset初始化失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _load_eeg_data(self, data_dir):
        """加载EEG数据"""
        if self.debug:
            print(f"正在加载EEG数据")
            print(f"  数据目录: {data_dir}")
            print(f"  JSON路径: {self.json_path}")
            print(f"  最大文件数: {self.max_files}")
            print(f"  固定采样率: {self.original_fs}Hz -> {self.target_fs}Hz")
            print(f"  下采样因子: {self.original_fs/self.target_fs:.1f}")
            # 从self.args中获取subject_ids参数
        subject_ids = getattr(self.args, 'subject_ids', None)
    
        if self.debug and subject_ids:
            print(f"  指定被试: {subject_ids}")
        # 检查数据目录
        if not os.path.exists(data_dir):
            print(f"❌ 数据目录不存在: {data_dir}")

            # 尝试查找替代路径
            print(f"🔍 尝试查找替代路径...")
            possible_paths = [
                data_dir,
                "/root/autodl-tmp/InterpretGatedNetwork-main/data",
                "/root/autodl-tmp/InterpretGatedNetwork-main/data/imagine",
                "/root/autodl-tmp/InterpretGatedNetwork-main/datasets",
                "/root/autodl-tmp/InterpretGatedNetwork-main"
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    print(f"  ✓ 找到存在路径: {path}")
                    # 检查是否是目录
                    if os.path.isdir(path):
                        # 列出内容
                        contents = os.listdir(path)[:10]  # 前10个
                        print(f"    目录内容: {contents}")
                    # 更新data_dir
                    data_dir = path
                    print(f"  → 使用路径: {data_dir}")
                    break

        if not os.path.exists(data_dir):
            return None

        # 检查JSON文件
        if not os.path.exists(self.json_path):
            print(f"❌ JSON文件不存在: {self.json_path}")
            # 尝试查找json文件
            json_candidates = [
                self.json_path,
                "/root/autodl-tmp/InterpretGatedNetwork-main/data/textmaps.json",
                "/root/autodl-tmp/InterpretGatedNetwork-main/datasets/textmaps.json",
                "/root/autodl-tmp/InterpretGatedNetwork-main/textmaps.json"
            ]

            for json_path in json_candidates:
                if os.path.exists(json_path):
                    print(f"  ✓ 找到JSON文件: {json_path}")
                    self.json_path = json_path
                    break

            if not os.path.exists(self.json_path):
                return None

        try:
            data_dict = process_imagine_fif_data_with_label_mapping(
                data_dir, 
                self.json_path, 
                self.max_files, 
                debug=self.debug,
                target_channels=self.target_channels,
                target_timepoints=self.target_timepoints,
                original_fs=self.original_fs,  # 使用固定的原始采样率
                target_fs=self.target_fs,  # 目标采样率
                downsample_method=self.downsample_method
            )

            if data_dict is None:
                print("❌ process_imagine_fif_data_with_label_mapping 返回 None")
                return None

            print(f"✓ 数据加载成功")
            print(f"  样本数: {len(data_dict.get('input_features', []))}")
            print(f"  实际形状: {data_dict.get('input_features', torch.tensor([])).shape}")

            return data_dict

        except Exception as e:
            print(f"❌ 调用process_imagine_fif_data_with_label_mapping失败: {e}")
            import traceback
            traceback.print_exc()
            return None


    
    def _setup_normalizer(self):
        """设置标准化器"""
        # EEG数据标准化策略：按样本标准化
        self.normalizer = Normalizer(norm_type='per_sample_std')
        
        # 如果有需要，可以预先计算全局统计量
        if 'input_features' in self.data_dict:
            all_features = self.data_dict['input_features']  # (n_samples, 122, 1651)
            
            # 转换为DataFrame格式以便使用Normalizer
            # 注意：这里需要将3D数据转换为2D格式
            n_samples, n_channels, n_times = all_features.shape
            features_2d = all_features.reshape(-1, n_times)  # (n_samples*n_channels, n_times)
            
            # 创建DataFrame
            df = pd.DataFrame(features_2d.numpy())
            
            # 计算标准化参数
            self.normalizer.normalize(df)
    
    def _normalize_sample(self, features):
        """对单个样本进行标准化"""
        if not self.scale or not hasattr(self, 'normalizer'):
            return features
        
        # 将特征转换为DataFrame
        n_channels, n_times = features.shape
        df = pd.DataFrame(features.numpy())
        
        # 使用Normalizer进行标准化
        normalized_df = self.normalizer.normalize(df)
        
        # 转换回tensor
        normalized_tensor = torch.tensor(normalized_df.values, dtype=torch.float32)
        
        return normalized_tensor
    
    '''def _split_samples_by_flag(self):
        """根据flag划分数据集"""
        all_samples = self._prepare_samples()
        n_samples = len(all_samples)
        
        if n_samples == 0:
            return []
        
        # 计算划分点
        n_test = int(n_samples * self.test_size)
        n_val = int(n_samples * self.val_size)
        n_train = n_samples - n_test - n_val
        
        # 打乱索引
        indices = np.random.permutation(n_samples)
        
        # 划分索引
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # 根据flag选择
        if self.flag == 'train':
            selected_indices = train_indices
        elif self.flag == 'val':
            selected_indices = val_indices
        elif self.flag == 'test':
            selected_indices = test_indices
        else:
            raise ValueError(f"无效的flag: {self.flag}")
        
        # 选择样本
        selected_samples = [all_samples[i] for i in selected_indices]
        
        if self.debug:
            print(f"\n数据划分:")
            print(f"  总共样本: {n_samples}")
            print(f"  Train: {len(train_indices)} 个")
            print(f"  Val:   {len(val_indices)} 个")
            print(f"  Test:  {len(test_indices)} 个")
            print(f"  {self.flag}: {len(selected_samples)} 个")
        
        return selected_samples'''
    def _split_samples_by_flag(self):
        """根据flag划分数据集"""
        all_samples = self._prepare_samples()
        n_samples = len(all_samples)
        print(f"\n数据划分:")
        print(f"  总样本数: {n_samples}")
        print(f"  测试集比例: {self.test_size}")
        print(f"  验证集比例: {self.val_size}")
        if n_samples == 0:
            if self.debug:
                print(f"⚠ 警告: 没有可用的样本")
            return []

        # 确保验证集比例不超过最大样本数
        n_val = int(n_samples * self.val_size)
        n_test = int(n_samples * self.test_size)
        n_train = n_samples - n_val - n_test

        # 确保每个分区至少有一个样本
        if n_train < 1:
            n_train = 1
            n_val = min(n_samples - 1, n_val)
            n_test = n_samples - n_train - n_val
        elif n_val < 1 and n_samples > 1:
            n_val = 1
            n_test = min(n_samples - n_train - 1, n_test)
            n_train = n_samples - n_val - n_test

        if self.debug:
            print(f"数据划分:")
            print(f"  总共样本: {n_samples}")
            print(f"  Train: {n_train} 个")
            print(f"  Val:   {n_val} 个")
            print(f"  Test:  {n_test} 个")

        # 打乱索引
        indices = np.random.permutation(n_samples)

        # 划分索引
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val] if n_val > 0 else []
        test_indices = indices[n_train + n_val:] if n_test > 0 else []

        # 根据flag选择
        if self.flag == 'train':
            selected_indices = train_indices
        elif self.flag == 'val':
            selected_indices = val_indices
        elif self.flag == 'test':
            selected_indices = test_indices
        else:
            raise ValueError(f"无效的flag: {self.flag}")

        if self.debug:
            print(f"  {self.flag}集: {len(selected_indices)} 个样本")

        # 选择样本
        selected_samples = [all_samples[i] for i in selected_indices] if len(selected_indices) > 0 else []

        return selected_samples
    
    def _prepare_samples(self):
        """准备样本列表"""
        samples = []
        if self.data_dict and 'input_features' in self.data_dict:
            input_features = self.data_dict['input_features']
            numeric_labels = self.data_dict['numeric_labels']
            text_labels = self.data_dict.get('text_labels', ['unknown'] * len(input_features))
            
            for i in range(len(input_features)):
                features = input_features[i]  # (122, 1651)
                label = numeric_labels[i]
                text_label = text_labels[i] if i < len(text_labels) else 'unknown'
                
                # 标准化
                if self.scale:
                    features = self._normalize_sample(features)
                else:
                    features = features.float()
                
                sample = {
                    'features': features,      # (122, 1651)
                    'label': torch.tensor(label, dtype=torch.long),  # 数字标签
                    'text_label': text_label  # 文本标签
                }
                samples.append(sample)
        
        return samples
    
    def __getitem__(self, index):
        """获取单个样本"""
        sample = self.samples[index]
        
        # 获取特征和标签
        seq_x = sample['features']  # (122, 1651)
        target = sample['label']    # 数字标签
        
        # 转置为框架期望的格式: (seq_len, feat_dim) = (1651, 122)
        # 这是为了与其他数据集保持一致
        seq_x = seq_x.transpose(0, 1)
        
        return seq_x, target
    
    def __len__(self):
        return len(self.samples)
    
    def inverse_transform(self, data):
        """逆变换（如需要）"""
        if not self.scale or not hasattr(self, 'normalizer'):
            return data
        # TODO: 实现逆标准化
        return data
    
    def get_class_distribution(self):
        """获取类别分布"""
        labels = [sample['label'].item() for sample in self.samples]
        label_counts = Counter(labels)
        
        distribution = {}
        for label_id, count in sorted(label_counts.items()):
            percentage = count / len(self.samples) * 100
            distribution[label_id] = {'count': count, 'percentage': percentage}
        
        return distribution
    
    def get_sample_info(self, index):
        """获取样本详细信息"""
        if index >= len(self.samples):
            raise IndexError(f"索引超出范围: {index}")
        
        sample = self.samples[index]
        return {
            'features_shape': sample['features'].shape,
            'label': sample['label'].item(),
            'text_label': sample['text_label'],
            'features_stats': {
                'min': sample['features'].min().item(),
                'max': sample['features'].max().item(),
                'mean': sample['features'].mean().item(),
                'std': sample['features'].std().item()
            }
        }


class EEGDataset3Class(EEGDataset):
    """3分类版本的EEG数据集"""
    def __init__(self, root_path, flag='train', size=None, features='S', 
                 data_path='', target='OT', scale=True, timeenc=0, freq='h', 
                 seasonal_patterns=None, nbins=10, bin_edges=None, 
                 json_path=None, max_files=10, debug=False, 
                 test_size=0.2, val_size=0.1, random_seed=42,subject_ids=None,args=None):
        
        # 先调用父类初始化
        super().__init__(
            root_path, flag, size, features, data_path, target, scale, 
            timeenc, freq, seasonal_patterns, nbins, bin_edges, 
            json_path, max_files, debug, test_size, val_size, random_seed
           
        )
        # 保存args
        self.args = args
        
        # 从args中获取subject_ids
        if subject_ids is None and args is not None:
            subject_ids = getattr(args, 'subject_ids', None)
        
        # 保存subject_ids参数
        self.subject_ids = subject_ids
        
        if debug and subject_ids:
            print(f"EEGDataset3Class: 处理被试: {subject_ids}")
        
        # 转换为3分类
        self._convert_to_3class()
        
        if self.debug:
            print(f"✓ EEGDataset3Class ({flag}集) 初始化完成")
            print(f"  样本数量: {len(self.samples)}")
            print(f"  类别数量: 3")
     # === 新增: 转换为DataFrame格式 ===
        #self._convert_to_dataframe_format()
       
        
    '''def _convert_to_dataframe_format(self):
        """将3D tensor数据转换为DataFrame格式"""
        import pandas as pd
        import numpy as np
        
        if not self.samples or not hasattr(self, 'data_dict'):
            return
        
        # 从samples重建tensor
        n_samples = len(self.samples)
        if n_samples == 0:
            return
        
        # 获取第一个样本的形状
        sample_feat = self.samples[0]['features']  # (122, 1651)
        n_channels, n_times = sample_feat.shape
        
        # 重建3D tensor
        all_features = np.zeros((n_samples, n_channels, n_times))
        all_labels = []
        
        for i, sample in enumerate(self.samples):
            all_features[i] = sample['features'].numpy()
            all_labels.append(sample['label'].item())
        
        # 转换为DataFrame格式
        rows = []
        for sample_idx in range(n_samples):
            for time_idx in range(n_times):
                row = {'sample_id': sample_idx, 'time_step': time_idx}
                for channel_idx in range(n_channels):
                    row[f'f_{channel_idx}'] = all_features[sample_idx, channel_idx, time_idx]
                rows.append(row)
        
        # 创建DataFrames
        self.feature_df = pd.DataFrame(rows)
        self.feature_df.set_index(['sample_id', 'time_step'], inplace=True)
        
        self.labels_df = pd.DataFrame({'label': all_labels})
        self.labels_df.index.name = 'sample_id'
        
        # 设置其他属性
        self.feature_names = self.feature_df.columns.tolist()
        self.all_IDs = list(range(n_samples))
        self.max_seq_len = n_times
        self.class_names = ["日常生活", "社交情感", "专业服务"]
        
        if self.debug:
            print(f"\n转换为DataFrame格式完成:")
            print(f"  feature_df形状: {self.feature_df.shape}")
            print(f"  labels_df形状: {self.labels_df.shape}")
            print(f"  特征维度: {len(self.feature_names)}")
            print(f"  样本数: {n_samples}")'''
    def _convert_to_3class(self):
        """将39类标签转换为3类"""
        # 使用 eeg_processor.py 中的函数
        mapping_3cat = create_3category_mapping()
        
        # 获取原始标签
        original_labels = [sample['label'].item() for sample in self.samples]
        unique_labels = set(original_labels)
        self.original_num_classes = len(unique_labels)
        
        if self.debug:
            print(f"\n转换前的标签分布 (39类):")
            label_counts = Counter(original_labels)
            for label_id in sorted(label_counts.keys()):
                count = label_counts[label_id]
                percentage = count / len(original_labels) * 100
                print(f"  类别{label_id}: {count} 样本 ({percentage:.1f}%)")
        
        # 转换每个样本的标签
        new_samples = []
        converted_count = 0
        for sample in self.samples:
            original_label = sample['label'].item()
            new_label = mapping_3cat.get(original_label, -1)
            
            if new_label >= 0:  # 只保留有效映射
                sample['label'] = torch.tensor(new_label, dtype=torch.long)
                new_samples.append(sample)
                converted_count += 1
            elif self.debug:
                print(f"警告: 标签{original_label} 无对应的3分类映射")
        
        # 更新样本列表
        self.samples = new_samples
        
        # 更新类别数为3
        self.num_classes = 3
        if hasattr(self, 'data_dict'):
            self.data_dict['num_classes'] = 3
        
        if self.debug:
            print(f"\n转换结果:")
            print(f"  原始样本数: {len(original_labels)}")
            print(f"  转换后样本数: {converted_count}")
            print(f"  有效转换率: {converted_count/len(original_labels)*100:.1f}%")
            print(f"  类别转换: {self.original_num_classes}类 -> 3类")
            
            # 显示3分类分布
            new_labels = [sample['label'].item() for sample in self.samples]
            label_counts_3cat = Counter(new_labels)
            label_names = {0: "日常生活", 1: "社交情感", 2: "专业服务"}
            print(f"\n3分类分布:")
            for label_id in sorted(label_counts_3cat.keys()):
                count = label_counts_3cat[label_id]
                percentage = count / len(self.samples) * 100
                name = label_names.get(label_id, f"未知类别{label_id}")
                print(f"  {name}({label_id}): {count} 样本 ({percentage:.1f}%)")
    
   
    """加载EEG数据，返回3分类"""
    def _load_eeg_data(self, data_dir):
        """加载EEG数据"""
        if self.debug:
            print(f"正在加载EEG数据")
            print(f"  数据目录: {data_dir}")
            print(f"  JSON路径: {self.json_path}")
            print(f"  最大文件数: {self.max_files}")
         # ✅ 从self中获取subject_ids参数
        if hasattr(self, 'subject_ids'):
            subject_ids = self.subject_ids
        else:
            subject_ids = None
            if self.debug:
                print(f"  ⚠ 警告: 没有subject_ids参数，将处理所有被试")

   

            # 检查数据目录
            if not os.path.exists(data_dir):
                print(f"❌ 数据目录不存在: {data_dir}")
                return None

            # 检查JSON文件
            if not os.path.exists(self.json_path):
                print(f"❌ JSON文件不存在: {self.json_path}")
                return None

            try:
                data_dict = process_imagine_fif_data_with_label_mapping(
                    data_dir, 
                    self.json_path, 
                    self.max_files, 
                    self.debug
                )

                if data_dict is None:
                    print("❌ process_imagine_fif_data_with_label_mapping 返回 None")
                    return None

                return data_dict

            except Exception as e:
                print(f"❌ 调用process_imagine_fif_data_with_label_mapping失败: {e}")
                import traceback
                traceback.print_exc()
                return None

            # 确保是3分类
            if 'num_classes' in data_dict and data_dict['num_classes'] != 3:
                if self.debug:
                    print(f"原始数据是 {data_dict['num_classes']} 类，正在转换为3分类...")

                # 将numeric_labels转换为3分类
                if 'numeric_labels' in data_dict:
                    original_labels = data_dict['numeric_labels']
                    # 转换为3分类
                    three_class_labels = convert_to_3category_labels(original_labels)

                    # 过滤有效样本
                    valid_indices = [i for i, label in enumerate(three_class_labels) if label >= 0]

                    if len(valid_indices) > 0:
                        data_dict['numeric_labels'] = [three_class_labels[i] for i in valid_indices]
                        data_dict['input_features'] = data_dict['input_features'][valid_indices]
                        if 'text_labels' in data_dict:
                            data_dict['text_labels'] = [data_dict['text_labels'][i] for i in valid_indices]

                        data_dict['num_classes'] = 3
                        data_dict['sample_count'] = len(data_dict['numeric_labels'])

                        if self.debug:
                            print(f"转换为3分类完成: {len(original_labels)} -> {len(valid_indices)} 个样本")
                    else:
                        raise ValueError("转换为3分类后无有效样本")

            return data_dict




