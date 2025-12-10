#!/usr/bin/env python3
"""
test_data_loader.py - 直接测试EEG数据加载函数
"""

import os
import sys
import json
import traceback
import numpy as np
from pathlib import Path

def main():
    """主函数"""
    # 添加项目路径
    project_root = Path("/root/autodl-tmp/InterpretGatedNetwork-main")
    sys.path.insert(0, str(project_root))
    
    print("=" * 60)
    print("直接测试 EEG 数据加载函数")
    print("=" * 60)
    
    # 1. 设置路径
    data_dir = "/root/autodl-tmp/InterpretGatedNetwork-main/data"
    json_path = "/root/autodl-tmp/json/textmaps.json"
    
    print(f"数据目录: {data_dir}")
    print(f"JSON路径: {json_path}")
    
    # 检查路径是否存在
    if not os.path.exists(data_dir):
        print(f"✗ 数据目录不存在: {data_dir}")
        # 列出根目录
        parent_dir = "/root/autodl-tmp/InterpretGatedNetwork-main"
        if os.path.exists(parent_dir):
            print("\n根目录内容:")
            for item in os.listdir(parent_dir):
                print(f"  - {item}")
        sys.exit(1)
    
    if not os.path.exists(json_path):
        print(f"✗ JSON文件不存在: {json_path}")
        # 查找json文件
        print("\n查找json文件...")
        for root, dirs, files in os.walk("/root/autodl-tmp"):
            for file in files:
                if file.endswith('textmaps.json'):
                    json_path = os.path.join(root, file)
                    print(f"✓ 找到JSON文件: {json_path}")
                    break
        if not os.path.exists(json_path):
            sys.exit(1)
    
    # 2. 检查FIF文件
    print("\n" + "="*60)
    print("查找FIF文件...")
    fif_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.fif', '.fif.gz')):
                fif_files.append(os.path.join(root, file))
    
    if not fif_files:
        print("✗ 未找到任何FIF文件")
        print("\ndata目录结构:")
        for root, dirs, files in os.walk(data_dir, topdown=True):
            level = root.replace(data_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # 只显示前5个文件
                if file.endswith('.fif') or file.endswith('.json'):
                    print(f'{subindent}{file}')
    else:
        print(f"✓ 找到 {len(fif_files)} 个FIF文件")
        print("\n前5个FIF文件:")
        for i, f in enumerate(fif_files[:5]):
            print(f"  {i+1}. {os.path.relpath(f, data_dir)}")
            size_mb = os.path.getsize(f) / (1024 * 1024)
            print(f"      大小: {size_mb:.1f} MB")
    
    # 3. 测试导入eeg_processor
    print("\n" + "="*60)
    print("导入 eeg_processor 模块...")
    try:
        from data_provider.eeg_processor import process_imagine_fif_data_with_label_mapping
        from data_provider.eeg_processor import (
            load_text_maps,
            find_imagine_fif_files,
            preprocess_eeg_data,
            validate_eeg_data
        )
        print("✓ 成功导入 eeg_processor 函数")
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        print("检查 eeg_processor.py 文件是否存在:")
        eeg_processor_path = os.path.join(project_root, "data_provider", "eeg_processor.py")
        if os.path.exists(eeg_processor_path):
            print(f"  ✓ 文件存在: {eeg_processor_path}")
            with open(eeg_processor_path, 'r') as f:
                first_lines = f.readlines()[:10]
            print(f"  文件开头内容:")
            for i, line in enumerate(first_lines, 1):
                print(f"  {i:2d}: {line.rstrip()}")
        else:
            print(f"  ✗ 文件不存在: {eeg_processor_path}")
        sys.exit(1)
    
    # 4. 测试单个函数
    print("\n" + "="*60)
    print("测试单个函数...")
    
    # 4.1 测试load_text_maps
    print("\n[1] 测试 load_text_maps:")
    try:
        text_maps = load_text_maps(json_path)
        if text_maps:
            print(f"✓ 成功加载textmaps.json")
            print(f"  包含 {len(text_maps)} 个映射")
            print(f"  前5个映射:")
            for i, (key, val) in enumerate(list(text_maps.items())[:5]):
                print(f"    {key}: {val}")
        else:
            print("✗ 加载JSON文件失败")
    except Exception as e:
        print(f"✗ load_text_maps失败: {e}")
        traceback.print_exc()
    
    # 4.2 测试find_imagine_fif_files
    print("\n[2] 测试 find_imagine_fif_files:")
    try:
        fif_files = find_imagine_fif_files(data_dir)
        if fif_files:
            print(f"✓ 找到 {len(fif_files)} 个imagine FIF文件")
            print(f"  前3个文件:")
            for i, f in enumerate(fif_files[:3]):
                print(f"    {i+1}. {os.path.basename(f)}")
        else:
            print("✗ 未找到imagine FIF文件")
            # 检查是否有其他FIF文件
            all_fif = []
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.endswith('.fif'):
                        all_fif.append(os.path.join(root, file))
            if all_fif:
                print(f"  但找到 {len(all_fif)} 个普通FIF文件")
                for f in all_fif[:3]:
                    print(f"    - {os.path.relpath(f, data_dir)}")
    except Exception as e:
        print(f"✗ find_imagine_fif_files失败: {e}")
        traceback.print_exc()
    
    # 5. 测试MNE
    print("\n" + "="*60)
    print("测试MNE模块...")
    try:
        import mne
        print(f"✓ MNE版本: {mne.__version__}")
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
        
        # 测试读取FIF文件
        if fif_files:
            test_file = fif_files[0]
            print(f"\n测试读取FIF文件: {os.path.basename(test_file)}")
            try:
                # 尝试读取epochs
                print(f"  读取epochs...")
                epochs = mne.read_epochs(test_file, preload=True, verbose=False)
                print(f"  ✓ 读取Epochs成功")
                print(f"  Epochs数: {len(epochs)}")
                print(f"  通道数: {len(epochs.ch_names)}")
                print(f"  时间点数: {len(epochs.times)}")
                print(f"  采样率: {epochs.info['sfreq']} Hz")
                
                if hasattr(epochs, 'metadata') and epochs.metadata is not None:
                    print(f"  Metadata列: {list(epochs.metadata.columns)}")
                    if 'Word' in epochs.metadata.columns:
                        print(f"  Word标签示例: {epochs.metadata['Word'].head().tolist()}")
                
                # 测试数据获取
                print(f"\n  获取数据...")
                data = epochs.get_data()
                print(f"  ✓ 获取数据成功")
                print(f"  数据形状: (epochs, channels, time) = {data.shape}")
                print(f"  数据类型: {data.dtype}")
                print(f"  数据范围: [{data.min():.4f}, {data.max():.4f}]")
                
            except Exception as e:
                print(f"✗ 读取FIF文件失败: {e}")
                traceback.print_exc()
        else:
            print("无FIF文件可供测试")
            
    except Exception as e:
        print(f"✗ MNE测试失败: {e}")
        traceback.print_exc()
    
    # 6. 测试主函数
    print("\n" + "="*60)
    print("测试 process_imagine_fif_data_with_label_mapping 函数...")
    print(f"参数:")
    print(f"  data_dir: {data_dir}")
    print(f"  json_path: {json_path}")
    print(f"  max_files: 1")
    print(f"  debug: True")
    
    try:
        # 调用主函数
        data_dict = process_imagine_fif_data_with_label_mapping(
            data_dir=data_dir,
            json_path=json_path,
            max_files=1,  # 只处理1个文件
            debug=True
        )
        
        if data_dict is None:
            print("✗ 函数返回 None")
        else:
            print("✓ 函数成功返回 data_dict")
            print(f"\n" + "="*60)
            print("数据集详细信息:")
            print("="*60)
            
            # 1. 基本信息
            print(f"1. 数据集基本信息:")
            print(f"  样本数量: {data_dict.get('sample_count', len(data_dict.get('input_features', [])))}")
            print(f"  类别数量: {data_dict.get('num_classes', '未指定')}")
            print(f"  特征维度: {data_dict.get('feature_dim', '未指定')}")
            print(f"  序列长度: {data_dict.get('seq_len', '未指定')}")
            
            # 2. 输入特征
            if 'input_features' in data_dict:
                features = data_dict['input_features']
                print(f"\n2. 输入特征 (input_features):")
                print(f"  类型: {type(features)}")
                print(f"  形状: {features.shape}")
                print(f"  数据类型: {features.dtype}")
                print(f"  内存占用: {features.element_size() * features.nelement() / 1024:.2f} KB")
                
                # 计算统计信息
                if isinstance(features, torch.Tensor):
                    features_np = features.numpy()
                else:
                    features_np = features
                
                print(f"\n  统计信息:")
                print(f"    最小值: {features_np.min():.6f}")
                print(f"    最大值: {features_np.max():.6f}")
                print(f"    平均值: {features_np.mean():.6f}")
                print(f"    标准差: {features_np.std():.6f}")
                print(f"    中位数: {np.median(features_np):.6f}")
                
                # 检查NaN和Inf
                nan_count = np.sum(np.isnan(features_np))
                inf_count = np.sum(np.isinf(features_np))
                print(f"    NaN数量: {nan_count}")
                print(f"    Inf数量: {inf_count}")
                
                if nan_count > 0 or inf_count > 0:
                    print(f"  ⚠ 警告: 数据包含NaN或Inf值!")
                
                # 检查数据范围
                if features_np.min() < -1000 or features_np.max() > 1000:
                    print(f"  ⚠ 警告: 数据范围异常: [{features_np.min():.2f}, {features_np.max():.2f}]")
            
            # 3. 标签
            if 'numeric_labels' in data_dict:
                labels = data_dict['numeric_labels']
                print(f"\n3. 数字标签 (numeric_labels):")
                print(f"  类型: {type(labels)}")
                
                if isinstance(labels, torch.Tensor):
                    labels_np = labels.numpy()
                    print(f"  形状: {labels.shape}")
                    print(f"  数据类型: {labels.dtype}")
                else:
                    labels_np = np.array(labels)
                    print(f"  长度: {len(labels)}")
                    print(f"  数据类型: {labels_np.dtype}")
                
                # 标签统计
                unique_labels, counts = np.unique(labels_np, return_counts=True)
                print(f"\n  标签统计:")
                for label, count in zip(unique_labels, counts):
                    percentage = 100 * count / len(labels_np)
                    print(f"    标签 {label}: {count} 样本 ({percentage:.1f}%)")
                
                if len(unique_labels) > 10:
                    print(f"  ⚠ 警告: 标签数量较多 ({len(unique_labels)} 种)，可能需要进行标签转换")
            
            # 4. 文本标签
            if 'text_labels' in data_dict:
                text_labels = data_dict['text_labels']
                print(f"\n4. 文本标签 (text_labels):")
                print(f"  数量: {len(text_labels)}")
                if len(text_labels) > 0:
                    print(f"  前5个标签: {text_labels[:5]}")
                
                # 统计唯一标签
                unique_text = set(text_labels)
                print(f"  唯一文本标签: {len(unique_text)} 种")
                if len(unique_text) <= 10:
                    print(f"  具体标签: {list(unique_text)}")
            
            # 5. 注意力掩码
            if 'attention_mask' in data_dict:
                attention_mask = data_dict['attention_mask']
                print(f"\n5. 注意力掩码 (attention_mask):")
                print(f"  类型: {type(attention_mask)}")
                if isinstance(attention_mask, torch.Tensor):
                    print(f"  形状: {attention_mask.shape}")
                    print(f"  数据类型: {attention_mask.dtype}")
                    # 检查是否全是1
                    unique_values = torch.unique(attention_mask)
                    print(f"  唯一值: {unique_values.tolist()}")
            
            # 6. 数据验证
            print(f"\n" + "="*60)
            print("6. 数据一致性检查:")
            print("="*60)
            
            # 检查样本数量是否一致
            features_len = len(data_dict.get('input_features', []))
            labels_len = len(data_dict.get('numeric_labels', []))
            text_len = len(data_dict.get('text_labels', []))
            
            print(f"  输入特征样本数: {features_len}")
            print(f"  数字标签样本数: {labels_len}")
            print(f"  文本标签样本数: {text_len}")
            
            if features_len == labels_len == text_len:
                print(f"  ✓ 所有样本数量一致")
            else:
                print(f"  ✗ 样本数量不一致!")
                if features_len != labels_len:
                    print(f"    ✗ 输入特征({features_len}) != 数字标签({labels_len})")
                if labels_len != text_len:
                    print(f"    ✗ 数字标签({labels_len}) != 文本标签({text_len})")
            
            # 检查标签范围
            if 'numeric_labels' in data_dict:
                labels_np = np.array(data_dict['numeric_labels'])
                label_min = labels_np.min()
                label_max = labels_np.max()
                print(f"\n  标签范围: [{label_min}, {label_max}]")
                
                if label_min < 0 or label_max > 100:
                    print(f"  ⚠ 警告: 标签范围异常，可能需要检查标签映射")
            
            # 7. 示例数据
            print(f"\n" + "="*60)
            print("7. 前5个样本示例:")
            print("="*60)
            
            for i in range(min(5, features_len)):
                print(f"\n  样本 {i}:")
                if 'input_features' in data_dict:
                    feat = data_dict['input_features'][i] if hasattr(data_dict['input_features'], '__getitem__') else None
                    if feat is not None:
                        if isinstance(feat, torch.Tensor):
                            feat = feat.numpy()
                        print(f"    特征形状: {feat.shape}")
                        print(f"    特征统计: 范围[{feat.min():.4f}, {feat.max():.4f}], 均值{feat.mean():.4f}")
                
                if 'numeric_labels' in data_dict:
                    label = data_dict['numeric_labels'][i] if hasattr(data_dict['numeric_labels'], '__getitem__') else None
                    if label is not None:
                        print(f"    数字标签: {label}")
                
                if 'text_labels' in data_dict and i < len(data_dict['text_labels']):
                    text_label = data_dict['text_labels'][i]
                    print(f"    文本标签: {text_label}")
                
            print(f"\n" + "="*60)
            print("测试完成!")
            print("="*60)
            
    except Exception as e:
        print(f"✗ 函数执行失败: {e}")
        traceback.print_exc()
        print(f"\n详细错误信息:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()