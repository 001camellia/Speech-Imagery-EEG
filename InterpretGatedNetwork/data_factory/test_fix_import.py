#!/usr/bin/env python3
"""
fix_import_issues.py - 修复导入问题
"""

import os
import sys
import subprocess
import traceback

def fix_import_paths():
    """修复Python路径"""
    project_root = "/root/autodl-tmp/InterpretGatedNetwork-main"
    
    # 添加项目根目录
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"✅ 添加项目根目录: {project_root}")
    
    # 添加data_provider目录
    data_provider_dir = os.path.join(project_root, "data_provider")
    if data_provider_dir not in sys.path:
        sys.path.insert(0, data_provider_dir)
        print(f"✅ 添加data_provider目录: {data_provider_dir}")
    
    # 检查Python路径
    print("\n✅ 当前Python路径:")
    for i, path in enumerate(sys.path[:5]):  # 只显示前5个
        print(f"  {i+1}. {path}")
    
    # 检查文件存在
    print(f"\n✅ 检查重要文件:")
    important_files = [
        ("eeg_processor.py", os.path.join(data_provider_dir, "eeg_processor.py")),
        ("eeg.py", os.path.join(data_provider_dir, "eeg.py")),
        ("data_loader.py", os.path.join(data_provider_dir, "data_loader.py")),
        ("run.py", os.path.join(project_root, "run.py")),
    ]
    
    for name, path in important_files:
        if os.path.exists(path):
            print(f"  ✓ {name}: 存在 ({path})")
        else:
            print(f"  ✗ {name}: 不存在")
    
    return data_provider_dir, project_root

def test_imports():
    """测试导入"""
    print("\n" + "="*60)
    print("测试导入模块...")
    print("="*60)
    
    # 1. 测试导入eeg_processor
    print("\n1. 测试导入 eeg_processor:")
    try:
        from data_provider.eeg_processor import process_imagine_fif_data_with_label_mapping
        print("✅ 成功导入 process_imagine_fif_data_with_label_mapping")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("尝试直接导入...")
        
        # 直接导入
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "eeg_processor", 
            "/root/autodl-tmp/InterpretGatedNetwork-main/data_provider/eeg_processor.py"
        )
        if spec is not None:
            eeg_processor = importlib.util.module_from_spec(spec)
            sys.modules["eeg_processor"] = eeg_processor
            spec.loader.exec_module(eeg_processor)
            print("✅ 直接导入成功")
        else:
            print("❌ 无法加载eeg_processor模块")
    
    # 2. 测试导入eeg
    print("\n2. 测试导入 eeg:")
    try:
        from data_provider.eeg import EEGDataset, EEGDataset3Class, eeg_collate_fn
        print("✅ 成功导入 EEGDataset, EEGDataset3Class, eeg_collate_fn")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        traceback.print_exc()
    
    # 3. 测试导入data_loader
    print("\n3. 测试导入 data_loader:")
    try:
        from data_provider.data_loader import data_provider
        print("✅ 成功导入 data_provider")
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        traceback.print_exc()

def main():
    """主函数"""
    print("="*60)
    print("修复导入路径问题")
    print("="*60)
    
    # 修复路径
    data_provider_dir, project_root = fix_import_paths()
    
    # 测试导入
    test_imports()
    
    # 运行测试
    print("\n" + "="*60)
    print("运行数据加载测试...")
    print("="*60)
    
    # 创建测试脚本
    test_script = """
import os
import sys
import json
import traceback
import numpy as np
import torch

# 设置路径
project_root = "/root/autodl-tmp/InterpretGatedNetwork-main"
data_provider_dir = os.path.join(project_root, "data_provider")

# 添加路径
sys.path.insert(0, project_root)
sys.path.insert(0, data_provider_dir)

print("\\n=== 测试数据加载 ===")
print(f"工作目录: {os.getcwd()}")
print(f"Python路径:")
for i, p in enumerate(sys.path[:3]):
    print(f"  {i+1}. {p}")

try:
    # 测试导入eeg_processor
    print("\\n1. 导入eeg_processor...")
    from data_provider.eeg_processor import process_imagine_fif_data_with_label_mapping
    print("✅ 导入成功")
    
    # 设置路径
    data_dir = "/root/autodl-tmp/InterpretGatedNetwork-main/data"
    json_path = "/root/autodl-tmp/json/textmaps.json"
    
    print(f"\\n2. 检查路径...")
    print(f"  数据目录: {data_dir} - 存在: {os.path.exists(data_dir)}")
    print(f"  JSON路径: {json_path} - 存在: {os.path.exists(json_path)}")
    
    if os.path.exists(data_dir):
        print(f"  数据目录内容:")
        for item in os.listdir(data_dir)[:5]:
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                print(f"    - {item}/")
            else:
                print(f"    - {item}")
    
    # 3. 调用函数
    print("\\n3. 调用处理函数...")
    data_dict = process_imagine_fif_data_with_label_mapping(
        data_dir=data_dir,
        json_path=json_path,
        max_files=1,
        debug=True
    )
    
    if data_dict is not None:
        print(f"\\n✅ 数据加载成功!")
        print(f"返回的keys: {list(data_dict.keys())}")
        
        for key, value in data_dict.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, (list, tuple)):
                print(f"  {key}: 类型={type(value).__name__}, 长度={len(value)}")
            else:
                print(f"  {key}: 类型={type(value).__name__}")
        
        if 'input_features' in data_dict:
            print(f"\\n✅ 输入特征: {data_dict['input_features'].shape}")
        if 'numeric_labels' in data_dict:
            print(f"✅ 标签: {data_dict['numeric_labels'].shape}")
    else:
        print("\\n❌ 函数返回 None")
        
except Exception as e:
    print(f"\\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
"""
    
    # 写入并运行测试脚本
    test_file = os.path.join(project_root, "test_fix_import.py")
    with open(test_file, 'w') as f:
        f.write(test_script)
    
    print(f"✅ 创建测试脚本: {test_file}")
    print(f"运行测试脚本...")
    os.chdir(project_root)
    subprocess.run([sys.executable, "test_fix_import.py"])
    
    print("\n" + "="*60)
    print("完成!")
    print("="*60)

if __name__ == "__main__":
    main()