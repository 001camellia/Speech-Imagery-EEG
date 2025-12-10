import argparse
import copy
import torch
from exp.experiment_classification import Experiment as ClassificationExperiment
from exp.experiment_regression import Experiment as RegressionExperiment
import random
import numpy as np
import os
exp_dict = {
    "classification": ClassificationExperiment,
    "regression": RegressionExperiment
}

def get_args():
    parser = argparse.ArgumentParser()
    # ===== EEG 数据特定参数 =====
    parser.add_argument("--data", type=str, default="EEG3", 
                        choices=['EEG', 'EEG3', 'UEA'])
    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/InterpretGatedNetwork-main/data/preprocessed_fif",
                        help="EEG数据根目录")
    parser.add_argument("--json_path", type=str, default="/root/autodl-tmp/InterpretGatedNetwork-main/json/textmaps.json",
                    help="textmaps.json映射文件路径")
    print("1")
    parser.add_argument("--target_channels", type=int, default=122,
                        help="目标通道数，通常为122")
    parser.add_argument("--target_timepoints", type=int, default=1651,
                        help="目标时间点数，通常为1651")
    parser.add_argument("--max_files", type=int, default=1000,
                        help="最大处理的FIF文件数量")
    
    parser.add_argument("--max_subjects", type=int, default=5,
                        help="最大处理的受试者数量")
    
    parser.add_argument("--subject_id", type=str, default="sub-01",
                        help="选择要处理的受试者ID，如sub-01, sub-02")
    parser.add_argument("--subject_ids", type=str, nargs='+', default=["sub-01,sub-02,sub-03"],
                        help="选择要处理的受试者ID列表，如: sub-01 sub-02 sub-03")##--------虽然这里设置了参数 但是写了辅助函数一次性获得所有被试
    parser.add_argument("--task_type", type=str, default="imagine", 
                        choices=['imagine', 'read', 'both'],
                        help="任务类型：想象(imagine)、阅读(read)或两者都包括(both)")
    # EEGCNN模型专用参数
    parser.add_argument("--eegcnn_layers", type=int, default=2,
                        help="Transformer层数，0表示不使用Transformer")
    parser.add_argument("--eegcnn_pooling", type=str, default='mean',
                        choices=[None, 'mean', 'sum', 'top'], 
                        help="池化方式: None/mean/sum/top")
    parser.add_argument("--eegcnn_cnn_f1", type=int, default=8,
                        help="CNN第一层滤波器数")
    parser.add_argument("--eegcnn_cnn_f2", type=int, default=8,
                        help="CNN深度系数")
    parser.add_argument("--eegcnn_kernel1", type=int, default=125,
                        help="第一层卷积核长度")
    parser.add_argument("--eegcnn_kernel2", type=int, default=25,
                        help="第二层卷积核长度")
    parser.add_argument("--eegcnn_pool1", type=int, default=2,
                        help="第一层池化大小")
    parser.add_argument("--eegcnn_pool2", type=int, default=5,
                        help="第二层池化大小")
    parser.add_argument("--eegcnn_dropout1", type=float, default=0.1,
                        help="CNN dropout率")
    parser.add_argument("--eegcnn_dropout2", type=float, default=0.1,
                        help="Transformer dropout率")
    parser.add_argument("--eegcnn_n_heads", type=int, default=8,
                        help="Transformer注意力头数")
    parser.add_argument("--eegcnn_d_ff", type=int, default=256,
                        help="Transformer前馈网络维度")
    # SBM and InterpGN model hyperparameters
    #parser.add_argument("--data", type=str, default="UEA", choices=['UEA', 'Monash'])
    #parser.add_argument("--data_root", type=str, default="./data/UEA_multivariate")
    #parser.add_argument("--model", type=str, default='SBM', choices=['SBM', 'LTS', 'InterpGN', 'DNN','EEGCNN'])
    parser.add_argument("--model", type=str, default='InterpGN', choices=['SBM', 'LTS', 'InterpGN', 'DNN','EEGCNN'])
    #parser.add_argument("--dnn_type", type=str, default='FCN', choices=['FCN', 'Transformer', 'TimesNet', 'PatchTST', 'ResNet'])
    parser.add_argument("--dnn_type", type=str, default='Transformer', choices=['FCN', 'Transformer', 'TimesNet', 'PatchTST', 'ResNet'])
    parser.add_argument("--dataset", type=str, default="BasicMotions")
    parser.add_argument("--lambda_reg", type=float, default=0.1)
    parser.add_argument("--lambda_div", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=1.)
    parser.add_argument("--num_shapelet", type=int, default=10)
    parser.add_argument("--gating_value", type=float, default=None)
    parser.add_argument("--pos_weight", action="store_true")
    parser.add_argument("--sbm_cls", type=str, default='linear')
    parser.add_argument("--distance_func", type=str, default='euclidean')
    parser.add_argument("--beta_schedule", type=str, default='constant')
    parser.add_argument("--memory_efficient", action="store_true")

    # Experiment config
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--lr_decay", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_clip", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument("--min_epochs", type=int, default=0)
    parser.add_argument("--train_epochs", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--multi_gpu", action='store_true')
    parser.add_argument("--test_only", action='store_true')
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--amp", action='store_false', default=True)

    # basic config
    parser.add_argument('--task_name', type=str, default='classification',
                        help='task name, options:[classification, regression]')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    
    # DNN model configs
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of ff layers')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # TimesNet specific
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    args = parser.parse_args()
    # 改为：
    if args.data in ['EEG', 'EEG3']:
        args.root_path = args.data_root
    else:
        args.root_path = f"{args.data_root}/{args.dataset}"
    args.is_training = True
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
'''  1651长度
    if __name__ == "__main__":
    args = get_args()
    exp_cls = exp_dict[args.task_name]
    import os
    
    # 正确的EEG数据路径
    correct_eeg_path = "/root/autodl-tmp/InterpretGatedNetwork-main/data/preprocessed_fif"
    
    if args.data in ['EEG', 'EEG3']:
        # 覆盖参数
        args.data_path = correct_eeg_path
        args.root_path = correct_eeg_path
        
        print(f"\n{'='*50}")
        print("EEG路径修正:")
        print(f"{'='*50}")
        print(f"使用EEG数据路径: {args.data_path}")
        print(f"路径存在: {os.path.exists(args.data_path)}")
        
        if not os.path.exists(args.data_path):
            print(f"❌ 错误: 数据路径不存在!")
            print(f"请检查路径: {args.data_path}")
            print(f"\n可用的数据目录:")
            base_dir = "/root/autodl-tmp/InterpretGatedNetwork-main"
            for root, dirs, files in os.walk(base_dir):
                for dir_name in dirs:
                    if "fif" in dir_name.lower() or "eeg" in dir_name.lower():
                        print(f"  - {os.path.join(root, dir_name)}")
            exit(1)
                # 自动计算seq_len
        original_timepoints = 1651
        downsample_factor = args.original_fs / args.target_fs
        args.seq_len = int(original_timepoints / downsample_factor)
        print(f"\nEEG数据参数:")
        print(f"  原始采样率: {args.original_fs}Hz")
        print(f"  目标采样率: {args.target_fs}Hz")
        print(f"  下采样因子: {downsample_factor:.2f}")
        print(f"  序列长度: {args.seq_len}")
    # 设置随机种子
    random_seeds = [0, 42, 1234, 8237, 2023] if args.seed == -1 else [args.seed]
    
    for i, seed in enumerate(random_seeds):
        set_seed(seed)
        args.seed = seed
        
        print(f"\n{'='*50}")
        print(f"{'='*5} 实验 {i+1}/{len(random_seeds)} - 随机种子: {seed} {'='*5}")
        print(f"{'='*50}")
        
        # 创建实验实例
        experiment = exp_cls(args=args)
        experiment.print_args()
        print()
        
        # 检查模型检查点
        checkpoint_path = f"{experiment.checkpoint_dir}/checkpoint.pth"
        if not args.test_only:
            if os.path.exists(checkpoint_path):
                print(f"检查点已存在: {checkpoint_path}")
                print("跳过训练，直接加载模型进行测试...")
                experiment.model.load_state_dict(torch.load(checkpoint_path))
            else:
                print(f"{'='*5} 开始训练 {'='*5}")
                experiment.train()
                print(f"{'='*5} 训练完成 {'='*5}")
                torch.cuda.empty_cache()
                print()
        else:
            print(f"{'='*5} 仅测试模式 {'='*5}")
            if not os.path.exists(checkpoint_path):
                print(f"警告: 检查点不存在: {checkpoint_path}")
                print("无法进行测试，请先训练模型或提供正确的检查点")
                continue
        
        # 加载模型
        if os.path.exists(checkpoint_path):
            print(f"加载模型: {checkpoint_path}")
            experiment.model.load_state_dict(torch.load(checkpoint_path))
        else:
            print(f"警告: 检查点不存在: {checkpoint_path}")
            print("使用随机初始化模型进行测试")
        
        # 测试
        print(f"\n{'='*5} 测试 {'='*5}")
        try:
            test_loss, test_metrics, test_df = experiment.test(
                save_csv=True,
                result_dir=f"./result/{args.model}"
            )
            
            # 保存测试结果
            import pickle
            result_dir = os.path.dirname(checkpoint_path)
            result_file = f"{result_dir}/test_results.pkl"
            with open(result_file, 'wb') as f:
                pickle.dump({
                    'test_loss': test_loss,
                    'test_metrics': test_metrics,
                    'test_df': test_df,
                    'args': vars(args)
                }, f)
            
            print(f"测试结果已保存: {result_file}")
            
            # 打印测试结果
            print(f"测试结果:")
            print(f"  Loss: {test_loss:.4f}")
            if test_metrics:
                for key, value in test_metrics.items():
                    print(f"  {key}: {value:.4f}")
                    
        except Exception as e:
            print(f"测试过程中出错: {e}")
            import traceback
            traceback.print_exc()
        
        print()
        torch.cuda.empty_cache()'''

'''12.7:18:00
if __name__ == "__main__":
    args = get_args()
    exp_cls = exp_dict[args.task_name]
    import os
    
    # 正确的EEG数据路径
    correct_eeg_path = "/root/autodl-tmp/InterpretGatedNetwork-main/data/preprocessed_fif"
    
    if args.data in ['EEG', 'EEG3']:
        # 处理多个被试ID
        if hasattr(args, 'subject_ids') and args.subject_ids:
            # 使用新的subject_ids参数
            if isinstance(args.subject_ids, str):
                # 如果是字符串，按逗号分割
                subject_ids = [s.strip() for s in args.subject_ids.split(',')]
            else:
                # 如果是列表
                subject_ids = args.subject_ids
        else:
            # 向后兼容，使用单个subject_id
            subject_ids = [args.subject_id]
        
        # 确保subject_ids是列表
        if not isinstance(subject_ids, list):
            subject_ids = [subject_ids]
        
        # 将subject_ids保存到args中
        args.subject_ids = subject_ids
        
        args.data_path = correct_eeg_path
        args.root_path = correct_eeg_path
        
        print(f"\n{'='*50}")
        print("EEG路径修正:")
        print(f"{'='*50}")
        print(f"使用EEG数据路径: {args.data_path}")
        print(f"路径存在: {os.path.exists(args.data_path)}")
        print(f"处理的被试ID: {subject_ids}")
        print(f"被试数量: {len(subject_ids)}")
        
        if not os.path.exists(args.data_path):
            print(f"❌ 错误: 数据路径不存在!")
            print(f"请检查路径: {args.data_path}")
            print(f"\n可用的数据目录:")
            base_dir = "/root/autodl-tmp/InterpretGatedNetwork-main"
            for root, dirs, files in os.walk(base_dir):
                for dir_name in dirs:
                    if "fif" in dir_name.lower() or "eeg" in dir_name.lower():
                        print(f"  - {os.path.join(root, dir_name)}")
            exit(1)
        
        # 从EEGDataset中读取固定的采样率参数
        original_fs = 500
        target_fs = 256
        original_timepoints = 1651
        
        # 计算序列长度
        downsample_factor = original_fs / target_fs
        calculated_seq_len = int(original_timepoints / downsample_factor)
        
        print(f"\nEEG数据参数:")
        print(f"  原始采样率: {original_fs}Hz")
        print(f"  目标采样率: {target_fs}Hz")
        print(f"  原始时间点: {original_timepoints}")
        print(f"  计算的目标时间点: {calculated_seq_len}")
        print(f"  下采样因子: {downsample_factor:.2f}")
        print(f"  被试ID列表: {subject_ids}")
    
    # 设置随机种子
    random_seeds = [0, 42, 1234, 8237, 2023] if args.seed == -1 else [args.seed]
    
    for i, seed in enumerate(random_seeds):
        set_seed(seed)
        args.seed = seed
        
        print(f"\n{'='*50}")
        print(f"{'='*5} 实验 {i+1}/{len(random_seeds)} - 随机种子: {seed} {'='*5}")
        print(f"{'='*50}")
        
        # 创建实验实例
        experiment = exp_cls(args=args)
        experiment.print_args()
        print()
        
        # 检查模型检查点
        checkpoint_path = f"{experiment.checkpoint_dir}/checkpoint.pth"
        if not args.test_only:
            if os.path.exists(checkpoint_path):
                print(f"检查点已存在: {checkpoint_path}")
                print("跳过训练，直接加载模型进行测试...")
                experiment.model.load_state_dict(torch.load(checkpoint_path))
            else:
                print(f"{'='*5} 开始训练 {'='*5}")
                experiment.train()
                print(f"{'='*5} 训练完成 {'='*5}")
                torch.cuda.empty_cache()
                print()
        else:
            print(f"{'='*5} 仅测试模式 {'='*5}")
            if not os.path.exists(checkpoint_path):
                print(f"警告: 检查点不存在: {checkpoint_path}")
                print("无法进行测试，请先训练模型或提供正确的检查点")
                continue
        
        # 加载模型
        if os.path.exists(checkpoint_path):
            print(f"加载模型: {checkpoint_path}")
            experiment.model.load_state_dict(torch.load(checkpoint_path))
        else:
            print(f"警告: 检查点不存在: {checkpoint_path}")
            print("使用随机初始化模型进行测试")
        
        # 测试
        print(f"\n{'='*5} 测试 {'='*5}")
        try:
            test_loss, test_metrics, test_df = experiment.test(
                save_csv=True,
                result_dir=f"./result/{args.model}"
            )
            
            # 保存测试结果
            import pickle
            result_dir = os.path.dirname(checkpoint_path)
            result_file = f"{result_dir}/test_results.pkl"
            with open(result_file, 'wb') as f:
                pickle.dump({
                    'test_loss': test_loss,
                    'test_metrics': test_metrics,
                    'test_df': test_df,
                    'args': vars(args)
                }, f)
            
            print(f"测试结果已保存: {result_file}")
            
            # 打印测试结果
            print(f"\n{'='*50}")
            print("测试结果汇总")
            print(f"{'='*50}")
            
            # 打印Loss
            if isinstance(test_loss, (int, float)):
                print(f"Loss: {test_loss:.6f}")
            else:
                print(f"Loss: {test_loss}")
            
            # 打印其他指标
            if test_metrics and isinstance(test_metrics, dict):
                for key, value in test_metrics.items():
                    if value is not None:
                        if isinstance(value, dict):
                            print(f"\n{key}:")
                            for subkey, subvalue in value.items():
                                if isinstance(subvalue, (int, float)):
                                    if subkey in ['accuracy', 'loss', 'f1', 'precision', 'recall', 'auc']:
                                        print(f"  {subkey}: {subvalue:.4f}")
                                    else:
                                        print(f"  {subkey}: {subvalue}")
                                else:
                                    print(f"  {subkey}: {subvalue}")
                        elif isinstance(value, (int, float)):
                            if key in ['accuracy', 'loss', 'f1_score', 'precision', 'recall', 'auc']:
                                print(f"{key}: {value:.4f}")
                            else:
                                print(f"{key}: {value}")
                        else:
                            print(f"{key}: {value}")
            elif hasattr(test_metrics, 'accuracy'):
                # 处理ClassificationResult对象
                print(f"Accuracy: {test_metrics.accuracy:.4f}")
                print(f"Loss: {test_metrics.loss:.4f}")
                if hasattr(test_metrics, 'num_samples'):
                    print(f"Num Samples: {test_metrics.num_samples}")
            else:
                print(f"test_metrics类型: {type(test_metrics)}")
                print(f"test_metrics: {test_metrics}")
            
            # 与随机基线比较
            if test_metrics and isinstance(test_metrics, dict):
                if 'accuracy' in test_metrics and isinstance(test_metrics['accuracy'], (int, float)):
                    accuracy = test_metrics['accuracy']
                    num_classes = getattr(args, 'num_class', 3)
                    random_baseline = 100.0 / num_classes
                    improvement = accuracy - random_baseline
                    
                    print(f"\n性能分析:")
                    print(f"  模型准确率: {accuracy:.2f}%")
                    print(f"  随机基线 ({num_classes}分类): {random_baseline:.2f}%")
                    print(f"  提升: {improvement:+.2f}% ({improvement/random_baseline*100:.1f}%相对提升)")
                    
                    if improvement > 0:
                        print(f"  ✓ 模型优于随机猜测")
                    else:
                        print(f"  ⚠ 模型低于随机猜测")
                    
                    # 与图片中的模型比较
                    if args.data in ['EEG3'] and num_classes == 3:
                        print(f"\n与论文模型比较:")
                        print(f"  论文模型准确率: 13.83% (平均)")
                        print(f"  您的模型准确率: {accuracy:.2f}%")
                        print(f"  差异: {accuracy - 13.83:+.2f}%")
                        
                        if accuracy > 13.83:
                            print(f"  ✓ 优于论文模型 (+{accuracy - 13.83:.2f}%)")
                        else:
                            print(f"  ⚠ 低于论文模型 ({accuracy - 13.83:.2f}%)")
            
            print(f"{'='*50}")
            
        except Exception as e:
            print(f"测试过程中出错: {e}")
            import traceback
            traceback.print_exc()
        
        print()
        torch.cuda.empty_cache()'''
if __name__ == "__main__":
    args = get_args()
    exp_cls = exp_dict[args.task_name]
    import os
    
    # 正确的EEG数据路径
    correct_eeg_path = "/root/autodl-tmp/InterpretGatedNetwork-main/data/preprocessed_fif"
    
    if args.data in ['EEG', 'EEG3']:
        # 处理多个被试ID
        if hasattr(args, 'subject_ids') and args.subject_ids:
            # 使用新的subject_ids参数
            if isinstance(args.subject_ids, str):
                # 如果是字符串，按逗号分割
                subject_ids = [s.strip() for s in args.subject_ids.split(',')]
            else:
                # 如果是列表
                subject_ids = args.subject_ids
        else:
            # 向后兼容，使用单个subject_id
            subject_ids = [args.subject_id]
        
        # 确保subject_ids是列表
        if not isinstance(subject_ids, list):
            subject_ids = [subject_ids]
        
        # 将subject_ids保存到args中
        args.subject_ids = subject_ids
        args = get_args()
    
        # 打印参数检查
        print(f"run_命令行参数:")
        print(f"  subject_ids: {args.subject_ids}")
        print(f"  subject_ids类型: {type(args.subject_ids)}")
        args.data_path = correct_eeg_path
        args.root_path = correct_eeg_path
        
        print(f"\n{'='*50}")
        print("EEG路径修正:")
        print(f"{'='*50}")
        print(f"使用EEG数据路径: {args.data_path}")
        print(f"路径存在: {os.path.exists(args.data_path)}")
        print(f"处理的被试ID: {subject_ids}")
        print(f"被试数量: {len(subject_ids)}")
        
        if not os.path.exists(args.data_path):
            print(f"❌ 错误: 数据路径不存在!")
            print(f"请检查路径: {args.data_path}")
            print(f"\n可用的数据目录:")
            base_dir = "/root/autodl-tmp/InterpretGatedNetwork-main"
            for root, dirs, files in os.walk(base_dir):
                for dir_name in dirs:
                    if "fif" in dir_name.lower() or "eeg" in dir_name.lower():
                        print(f"  - {os.path.join(root, dir_name)}")
            exit(1)
        
        # 从EEGDataset中读取固定的采样率参数
        original_fs = 500
        target_fs = 256
        original_timepoints = 1651
        
        # 计算序列长度
        downsample_factor = original_fs / target_fs
        calculated_seq_len = int(original_timepoints / downsample_factor)
        
        print(f"\nEEG数据参数:")
        print(f"  原始采样率: {original_fs}Hz")
        print(f"  目标采样率: {target_fs}Hz")
        print(f"  原始时间点: {original_timepoints}")
        print(f"  计算的目标时间点: {calculated_seq_len}")
        print(f"  下采样因子: {downsample_factor:.2f}")
        print(f"  被试ID列表: {subject_ids}")
    
    # 设置随机种子
    random_seeds = [0, 42, 1234, 8237, 2023] if args.seed == -1 else [args.seed]
    
    for i, seed in enumerate(random_seeds):
        set_seed(seed)
        args.seed = seed
        
        print(f"\n{'='*50}")
        print(f"{'='*5} 实验 {i+1}/{len(random_seeds)} - 随机种子: {seed} {'='*5}")
        print(f"{'='*50}")
        
        # 创建实验实例
        experiment = exp_cls(args=args)
        experiment.print_args()
        print()
        
        # 检查模型检查点
        checkpoint_path = f"{experiment.checkpoint_dir}/checkpoint.pth"
        if not args.test_only:
            if os.path.exists(checkpoint_path):
                print(f"检查点已存在: {checkpoint_path}")
                print("跳过训练，直接加载模型进行测试...")
                experiment.model.load_state_dict(torch.load(checkpoint_path))
            else:
                print(f"{'='*5} 开始训练 {'='*5}")
                experiment.train()
                print(f"{'='*5} 训练完成 {'='*5}")
                torch.cuda.empty_cache()
                print()
        else:
            print(f"{'='*5} 仅测试模式 {'='*5}")
            if not os.path.exists(checkpoint_path):
                print(f"警告: 检查点不存在: {checkpoint_path}")
                print("无法进行测试，请先训练模型或提供正确的检查点")
                continue
        
        # 加载模型
        if os.path.exists(checkpoint_path):
            print(f"加载模型: {checkpoint_path}")
            experiment.model.load_state_dict(torch.load(checkpoint_path))
        else:
            print(f"警告: 检查点不存在: {checkpoint_path}")
            print("使用随机初始化模型进行测试")
        
        # 测试
        print(f"\n{'='*5} 测试 {'='*5}")
        try:
            test_loss, test_metrics, test_df = experiment.test(
                save_csv=True,
                result_dir=f"./result/{args.model}"
            )
            
            # 保存测试结果
            import pickle
            result_dir = os.path.dirname(checkpoint_path)
            result_file = f"{result_dir}/test_results.pkl"
            with open(result_file, 'wb') as f:
                pickle.dump({
                    'test_loss': test_loss,
                    'test_metrics': test_metrics,
                    'test_df': test_df,
                    'args': vars(args)
                }, f)
            
            print(f"测试结果已保存: {result_file}")
            
            # 打印测试结果
            print(f"\n{'='*50}")
            print("测试结果汇总")
            print(f"{'='*50}")
            
            # 打印Loss
            if isinstance(test_loss, (int, float)):
                print(f"测试损失: {test_loss:.6f}")
            
            # 打印其他指标
            if test_metrics and isinstance(test_metrics, dict):
                for key, value in test_metrics.items():
                    if value is not None:
                        if isinstance(value, dict):
                            print(f"\n{key}:")
                            for subkey, subvalue in value.items():
                                if isinstance(subvalue, (int, float)):
                                    if subkey in ['accuracy', 'loss', 'f1', 'precision', 'recall', 'auc']:
                                        print(f"  {subkey}: {subvalue:.4f}")
                                    else:
                                        print(f"  {subkey}: {subvalue}")
                                else:
                                    print(f"  {subkey}: {subvalue}")
                        elif isinstance(value, (int, float)):
                            if key in ['accuracy', 'loss', 'f1_score', 'precision', 'recall', 'auc']:
                                print(f"{key}: {value:.4f}")
                            else:
                                print(f"{key}: {value}")
                        else:
                            print(f"{key}: {value}")
            elif hasattr(test_metrics, 'accuracy'):
                # 处理ClassificationResult对象
                print(f"准确率: {test_metrics.accuracy:.4f}")
                print(f"损失: {test_metrics.loss:.4f}")
                if hasattr(test_metrics, 'num_samples'):
                    print(f"样本数: {test_metrics.num_samples}")
            
            # 与随机基线比较
            if test_metrics and isinstance(test_metrics, dict) and 'accuracy' in test_metrics:
                accuracy = test_metrics['accuracy']
                if isinstance(accuracy, (int, float)):
                    num_classes = getattr(args, 'num_class', 3)
                    random_baseline = 100.0 / num_classes
                    improvement = accuracy - random_baseline
                    
                    print(f"\n性能分析:")
                    print(f"  模型准确率: {accuracy:.2f}%")
                    print(f"  随机基线 ({num_classes}分类): {random_baseline:.2f}%")
                    print(f"  提升: {improvement:+.2f}% ({improvement/random_baseline*100:.1f}%相对提升)")
                    
                    if improvement > 0:
                        print(f"  ✓ 模型优于随机猜测")
                    else:
                        print(f"  ⚠ 模型低于随机猜测")
            
            print(f"{'='*50}")
            
        except Exception as e:
            print(f"测试过程中出错: {e}")
            import traceback
            traceback.print_exc()
        
        print()
        torch.cuda.empty_cache()