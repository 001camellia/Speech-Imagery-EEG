import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import time
from datetime import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.metrics import accuracy_score
from models.Shapelet import ShapeBottleneckModel, DistThresholdSBM
from models.InterpGN import InterpGN, FullyConvNetwork, Transformer, TimesNet, PatchTST, ResNet
from utils.tools import EarlyStopping, convert_to_hms, gini_coefficient
from utils.shapelet_util import ClassificationResult
from data_provider.data_factory import data_provider
from models.eegcnn import EEGCNNTransformer

def compute_beta(epoch, max_epoch, schedule='cosine'):
    if schedule == 'cosine':
        beta = 1/2 * (1 + np.cos(np.pi*epoch/max_epoch))
    elif schedule == 'linear':
        beta = 1 - epoch/max_epoch
    else:
        beta = 1
    return beta


def compute_shapelet_score(shapelet_distances, cls_weights, y_pred, y_true):
    score = shapelet_distances @ nn.functional.relu(cls_weights.T) / shapelet_distances.shape[-1]
    score_correct = score[y_pred == y_true]
    class_correct = y_true[y_pred == y_true]
    score_class = score_correct.gather(-1, class_correct.unsqueeze(1))
    return score_class.mean().item()


def get_dnn_model(configs):
    dnn_dict = {
        'FCN': FullyConvNetwork,
        'Transformer': Transformer,
        'TimesNet': TimesNet,
        'PatchTST': PatchTST,
        'ResNet': ResNet
    }
    return dnn_dict[configs.dnn_type](configs)

# 首先修改模型字典
def get_eegcnn_model(configs):
    """EEGCNN模型工厂函数"""
    from models.model_eegcnn import EEGCNNTransformer
   
    
    # 打印调试信息
    print(f"[DEBUG] get_eegcnn_model 接收到的 configs:")
    print(f"  configs 类型: {type(configs)}")
    print(f"  configs 属性: {list(vars(configs).keys())[:20] if hasattr(configs, '__dict__') else '无'}")
    print(f"  enc_in: {getattr(configs, 'enc_in', '无')}")
    print(f"  seq_len: {getattr(configs, 'seq_len', '无')}")
    print(f"  c_out: {getattr(configs, 'c_out', '无')}")
    
    # 使用适配器包装
    from models.eegcnn import EEGCNNAdapter
    return EEGCNNAdapter(configs)
    # 使用适配器包装
    return EEGCNNAdapter(configs)
    '''return EEGCNNTransformer(
        input_channels=configs.enc_in,
        seq_len=configs.seq_len,
        num_classes=configs.c_out,
        dropout1=getattr(configs, 'eegcnn_dropout1', 0.1),
        dropout2=getattr(configs, 'eegcnn_dropout2', 0.1),
        num_layers=getattr(configs, 'eegcnn_layers', 0),
        pooling=getattr(configs, 'eegcnn_pooling', None),
        cnn_filter1=getattr(configs, 'eegcnn_cnn_f1', 8),
        cnn_filter2=getattr(configs, 'eegcnn_cnn_f2', 8),
        kernel_length1=getattr(configs, 'eegcnn_kernel1', 125),
        kernel_length2=getattr(configs, 'eegcnn_kernel2', 25),
        pooling1=getattr(configs, 'eegcnn_pool1', 2),
        pooling2=getattr(configs, 'eegcnn_pool2', 5),
        d_model=getattr(configs, 'd_model', 512),
        n_heads=getattr(configs, 'eegcnn_n_heads', 8),
        dim_feedforward=getattr(configs, 'eegcnn_d_ff', 256)
    )'''

class Experiment(object):
    # 类属性
    model_dict = {
        'InterpGN': InterpGN,
        'SBM': ShapeBottleneckModel,
        'LTS': DistThresholdSBM,
        'DNN': get_dnn_model,
        'EEGCNN': get_eegcnn_model
    }
    
    def __init__(self, args):
        """
        实验类初始化
        """
        # 保存参数
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*50}")
        print("Experiment初始化")
        print(f"{'='*50}")
        
        # 1. 加载数据
        print("1. 加载数据...")
        self.train_data, self.train_loader = data_provider(args, flag="train")
        self.val_data, self.val_loader = data_provider(args, flag="val")
        self.test_data, self.test_loader = data_provider(args, flag="test")
        
        # 打印数据信息
        print(f"✓ 数据加载完成")
        print(f"  训练集大小: {len(self.train_data) if hasattr(self.train_data, '__len__') else 'N/A'}")
        print(f"  验证集大小: {len(self.val_data) if hasattr(self.val_data, '__len__') else 'N/A'}")
        print(f"  测试集大小: {len(self.test_data) if hasattr(self.test_data, '__len__') else 'N/A'}")
        
        # 检查数据是否为空
        if (not hasattr(self.train_data, '__len__') or len(self.train_data) == 0) and (not args.test_only):
            print(f"⚠ 警告: 训练集为空!")
            print(f"  如果这是测试模式，请忽略此警告")
        
        # 2. 从数据集中获取实际形状
        print("\n2. 从数据中获取参数...")
        self._get_params_from_data()
        
        # 3. 构建模型
        print("\n3. 构建模型...")
        self.model = self._build_model().to(self.device)
        print(f"\n构建模型:")
        print(f"  模型类型: {self.args.model}")
        print(f"  DNN类型: {getattr(self.args, 'dnn_type', 'N/A')}")
       
        # 4. 设置优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=self.args.train_epochs
        )
        self.checkpoint_dir = "./checkpoints/{}/{}/dnn-{}_seed-{}_k-{}_div-{}_reg-{}_eps-{}_beta-{}_dfunc-{}_cls-{}".format(
            self.args.model,
            self.args.dataset,
            self.args.dnn_type,
            self.args.seed,
            self.args.num_shapelet,
            self.args.lambda_div,
            self.args.lambda_reg,
            self.args.epsilon,
            self.args.beta_schedule,
            self.args.distance_func,
            self.args.sbm_cls
        )
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        # 5. 设置损失函数
        self.loss_fn = nn.CrossEntropyLoss()
        self.epoch_stop = 0
        
        print(f"\n✓ Experiment 初始化完成")
        print(f"  模型: {self.args.model}")
        print(f"  设备: {self.device}")
        print(f"  检查点目录: {self.checkpoint_dir}")
        print(f"{'='*50}")
    
    def _get_params_from_data(self):
        """从数据集获取参数"""
        # 获取序列长度
        if hasattr(self.train_data, 'seq_len'):
            # 从数据集的属性获取
            self.args.seq_len = self.train_data.seq_len
        elif hasattr(self.train_data, 'samples') and len(self.train_data.samples) > 0:
            # 从第一个样本获取
            try:
                sample_x, _ = self.train_data[0]  # (seq_len, channels)
                self.args.seq_len = sample_x.shape[0]  # 转置后的形状
            except:
                self.args.seq_len = 845
        elif hasattr(self.train_data, 'data_dict') and 'input_features' in self.train_data.data_dict:
            # 从data_dict获取
            input_features = self.train_data.data_dict['input_features']
            self.args.seq_len = input_features.shape[2]  # (n_samples, n_channels, n_times)
        else:
            # 默认值
            self.args.seq_len = 845
            print(f"⚠ 警告: 无法从数据获取seq_len，使用默认值: {self.args.seq_len}")
        
        self.args.pred_len = 0
        self.args.label_len = 0
        
        # 获取特征维度
        if hasattr(self.train_data, 'enc_in'):
            # 从数据集获取
            self.args.enc_in = self.train_data.enc_in
        elif hasattr(self.train_data, 'target_channels'):
            self.args.enc_in = self.train_data.target_channels
        elif hasattr(self.train_data, 'samples') and len(self.train_data.samples) > 0:
            try:
                sample_x, _ = self.train_data[0]  # (seq_len, channels)
                self.args.enc_in = sample_x.shape[1]  # 通道数
            except:
                self.args.enc_in = 122
        elif hasattr(self.train_data, 'data_dict') and 'input_features' in self.train_data.data_dict:
            # 从data_dict获取
            input_features = self.train_data.data_dict['input_features']
            self.args.enc_in = input_features.shape[1]  # 通道数
        else:
            self.args.enc_in = 122
            print(f"⚠ 警告: 无法从数据获取enc_in，使用默认值: {self.args.enc_in}")
        
        # 获取类别数量
        if hasattr(self.train_data, 'num_classes'):
            self.args.num_class = self.train_data.num_classes
        elif hasattr(self.train_data, 'data_dict') and 'num_classes' in self.train_data.data_dict:
            self.args.num_class = self.train_data.data_dict['num_classes']
        elif hasattr(self.train_data, 'samples') and len(self.train_data.samples) > 0:
            # 统计前100个样本的标签
            try:
                labels = set()
                for i in range(min(100, len(self.train_data.samples))):
                    _, label = self.train_data[i]
                    labels.add(label.item())
                self.args.num_class = len(labels)
            except:
                self.args.num_class = 3
        else:
            self.args.num_class = 3
            print(f"⚠ 警告: 无法从数据获取num_class，使用默认值: {self.args.num_class}")
        
        # 获取采样率
        if hasattr(self.train_data, 'original_fs'):
            self.args.original_fs = self.train_data.original_fs
        else:
            self.args.original_fs = 500
            print(f"⚠ 警告: 无法从数据获取original_fs，使用默认值: {self.args.original_fs}Hz")
        
        if hasattr(self.train_data, 'target_fs'):
            self.args.target_fs = self.train_data.target_fs
        else:
            self.args.target_fs = 256
            print(f"⚠ 警告: 无法从数据获取target_fs，使用默认值: {self.args.target_fs}Hz")
        
        print(f"数据集配置 (从数据中获取):")
        print(f"  seq_len: {self.args.seq_len}")
        print(f"  enc_in: {self.args.enc_in}")
        print(f"  num_class: {self.args.num_class}")
        print(f"  original_fs: {self.args.original_fs}Hz")
        print(f"  target_fs: {self.args.target_fs}Hz")
        print(f"  data类型: {type(self.train_data)}")
    
    def _build_model(self):
        """构建模型"""
        shapelet_lengths = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
        num_shapelet = [self.args.num_shapelet] * len(shapelet_lengths)
        
        # 检查模型是否在字典中
        if self.args.model not in self.model_dict:
            raise ValueError(f"模型 '{self.args.model}' 不在可用模型字典中. 可用模型: {list(self.model_dict.keys())}")
        
        # 获取模型构造函数
        model_class = self.model_dict[self.args.model]
        
        # 构建模型
        if self.args.model in ['SBM', 'LTS']:
            model = model_class(
                configs=self.args,
                num_shapelet=num_shapelet,
                shapelet_len=shapelet_lengths,
            )
        elif self.args.model == 'DNN':
            model = get_dnn_model(self.args)
        elif self.args.model == 'EEGCNN':
            # 新增EEGCNN模型
            from models.eegcnn import EEGCNNTransformer
            model = EEGCNNTransformer(self.args)
        else:
            model = model_class(self.args)
        
        if self.args.multi_gpu:
            model = nn.DataParallel(model)
            print("  使用多GPU训练")
        
        return model
    
    def print_args(self):
        """打印参数"""
        print(f"\n{'='*50}")
        print("实验参数:")
        print(f"{'='*50}")
        for arg in vars(self.args):
            value = getattr(self.args, arg)
            print(f"  {arg}: {value}")
        print(f"{'='*50}")
    
    def train(self):
        """训练模型"""
        torch.set_float32_matmul_precision('medium')
        checkpoint_dir = self.checkpoint_dir
        time_start = time.time()
        
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, delta=0)
        train_step = 0
        
        for epoch in range(self.args.train_epochs):
            self.model.train()
            
            # 检查训练集是否为空
            if len(self.train_loader) == 0:
                print(f"Epoch {epoch+1}/{self.args.train_epochs} | 训练集为空，跳过训练")
                continue
                
            train_loss = []
            for i, (batch_x, label, padding_mask) in enumerate(self.train_loader):
                train_step += 1
                batch_x = batch_x.float().to(self.device)
                label = label.long().squeeze(-1).to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.args.amp):
                    if self.args.model == 'DNN':
                        logits = self.model(batch_x, padding_mask, None, None)
                        loss = nn.functional.cross_entropy(logits, label)
                    else:
                        logits, model_info = self.model(batch_x, padding_mask, None, None)
                        loss = nn.functional.cross_entropy(logits, label) + model_info.loss.mean()
                    
                    if self.args.model in ['InterpGN']:
                        beta = compute_beta(epoch, self.args.train_epochs, self.args.beta_schedule)
                        loss += beta * nn.functional.cross_entropy(model_info.shapelet_preds, label)
                
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                loss.backward()
                
                if train_step % self.args.gradient_accumulation_steps == 0:
                    if self.args.gradient_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.gradient_clip)
                    self.optimizer.step()
                    if self.args.pos_weight:
                        self.model.step()
                    self.optimizer.zero_grad()
                
                train_loss.append(loss.item())
            
            if not train_loss:
                print(f"Epoch {epoch+1}/{self.args.train_epochs} | 无有效训练数据")
                continue
                
            train_loss = np.mean(train_loss)
            val_loss, val_accuracy = self.validation()
            time_now = time.time()
            time_remain = (time_now - time_start) * (self.args.train_epochs - epoch) / (epoch + 1)
            
            if (epoch + 1) % self.args.log_interval == 0:
                print(f"Epoch {epoch+1}/{self.args.train_epochs} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Val Acc {val_accuracy:.4f} | Time Rem {convert_to_hms(time_remain)}")
            
            if self.args.lr_decay:
                self.scheduler.step()
            
            if epoch >= self.args.min_epochs:
                early_stopping(-val_accuracy, self.model, checkpoint_dir)
            
            if early_stopping.early_stop:
                print("Early stopping")
                self.epoch_stop = epoch
                break
            else:
                self.epoch_stop = epoch
            
            sys.stdout.flush()
        
        # 加载最佳模型
        best_model_path = checkpoint_dir + '/checkpoint.pth'
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            print(f"✓ 加载最佳模型: {best_model_path}")
        
        return self.model
    
    def validation(self):
        """验证模型"""
        # 检查验证集是否为空
        if len(self.val_loader) == 0:
            print("⚠ 警告: 验证集为空")
            return float('inf'), 0.0
            
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(self.val_loader):
                batch_x = batch_x.float().to(self.device)
                label = label.long().squeeze(-1).to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.args.amp):
                    if self.args.model == 'DNN':
                        logits = self.model(batch_x, padding_mask, None, None)
                        loss = nn.functional.cross_entropy(logits, label, reduction='none')
                    else:
                        logits, model_info = self.model(batch_x, padding_mask, None, None)
                        loss = nn.functional.cross_entropy(logits, label, reduction='none') + model_info.loss.mean()
                
                total_loss.append(loss.flatten())
                preds.append(logits.cpu())
                trues.append(label.cpu())
        
        if not total_loss:
            return float('inf'), 0.0
            
        total_loss = torch.cat(total_loss, dim=0).mean().item()
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        probs = torch.nn.functional.softmax(preds, dim=1)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        accuracy = accuracy_score(predictions, trues)
        
        self.model.train()
        return total_loss, accuracy
    
    '''def test(self, save_csv=True, result_dir=None):
        """测试模型"""
        if not os.path.isdir(result_dir):
            try:
                os.makedirs(result_dir)
            except:
                pass
        
        @dataclass
        class Buffer:
            x_data: list = field(default_factory=list)
            trues: list = field(default_factory=list)
            preds: list = field(default_factory=list)
            shapelet_preds: list = field(default_factory=list)
            dnn_preds: list = field(default_factory=list)
            p: list = field(default_factory=list)
            d: list = field(default_factory=list)
            eta: list = field(default_factory=list)
            loss: list = field(default_factory=list)
        
        buffer = Buffer()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(self.test_loader):
                batch_x = batch_x.float().to(self.device)
                label = label.long().squeeze(-1).to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.args.amp):
                    if self.args.model == 'DNN':
                        logits = self.model(batch_x, padding_mask, None, None)
                        loss = nn.functional.cross_entropy(logits, label, reduction='none')
                    else:
                        logits, model_info = self.model(batch_x, padding_mask, None, None, gating_value=self.args.gating_value)
                        loss = nn.functional.cross_entropy(logits, label, reduction='none') + model_info.loss.mean()
                
                buffer.loss.append(loss.flatten())
                buffer.x_data.append(batch_x.cpu())
                buffer.trues.append(label.cpu())
                buffer.preds.append(logits.cpu())
                
                if self.args.model != 'DNN':
                    buffer.p.append(model_info.p.cpu())
                    buffer.d.append(model_info.d.cpu())
                    buffer.shapelet_preds.append(model_info.shapelet_preds.cpu())
                    if self.args.model == 'InterpGN':
                        buffer.eta.append(model_info.eta.cpu())
                        buffer.dnn_preds.append(model_info.dnn_preds.cpu())
        
        probs = torch.nn.functional.softmax(torch.cat(buffer.preds, dim=0), dim=1)
        predictions = torch.argmax(probs, dim=1)
        trues = torch.cat(buffer.trues, dim=0).flatten()
        accuracy = accuracy_score(predictions.cpu().numpy(), trues.cpu().numpy())
        
        cls_result = ClassificationResult(
            x_data=torch.cat(buffer.x_data, dim=0).cpu(),
            trues=trues.cpu(),
            preds=predictions.cpu(),
            loss=torch.cat(buffer.loss, dim=0).mean().item(),
            accuracy=accuracy
        )
        
        if self.args.model != 'DNN':
            cls_result.p = torch.cat(buffer.p, dim=0).cpu()
            cls_result.d = torch.cat(buffer.d, dim=0).cpu()
            cls_result.shapelet_preds = torch.cat(buffer.shapelet_preds, dim=0).cpu()
            if self.args.model == 'InterpGN':
                cls_result.eta = torch.cat(buffer.eta, dim=0).cpu()
                cls_result.w = self.model.sbm.output_layer.weight.detach().cpu()
                cls_result.shapelets = self.model.sbm.get_shapelets()
            else:
                cls_result.w = self.model.output_layer.weight.detach().cpu()
                cls_result.shapelets = self.model.get_shapelets()
        
        test_loss = torch.cat(buffer.loss, dim=0).mean().item()
        test_df = None
        
        if save_csv:
            summary_dict = dict()
            for arg in vars(self.args):
                if arg in [
                    'model', 'dataset', 'dnn_type', 
                    'train_epochs', 'num_shapelet', 'lambda_reg', 'lambda_div', 'epsilon', 'lr', 
                    'seed', 'pos_weight', 'beta_schedule', 'gating_value',
                    'distance_func', 'sbm_cls'
                ]:
                    summary_dict[arg] = [getattr(self.args, arg)]
            
            summary_dict['test_accuracy'] = accuracy
            summary_dict['epoch_stop'] = self.epoch_stop
            
            if self.args.model != 'DNN':
                summary_dict['eta_mean'] = cls_result.eta.mean().cpu().item() if self.args.model == 'InterpGN' else None
                summary_dict['eta_std'] = cls_result.eta.std().cpu().item() if self.args.model == 'InterpGN' else None
                summary_dict['shapelet_score'] = compute_shapelet_score(cls_result.d, cls_result.w, cls_result.preds, cls_result.trues)
                summary_dict['w_count_1'] = (cls_result.w.abs() > 1).float().sum().item()
                summary_dict['w_ratio_1'] = (cls_result.w.abs() > 1).float().mean().item()
                summary_dict['w_count_0.5'] = (cls_result.w.abs() > 0.5).float().sum().item()
                summary_dict['w_ratio_0.5'] = (cls_result.w.abs() > 0.5).float().mean().item()
                summary_dict['w_count_0.1'] = (cls_result.w.abs() > 0.1).float().sum().item()
                summary_dict['w_ratio_0.1'] = (cls_result.w.abs() > 0.1).float().mean().item()
                summary_dict['w_max'] = cls_result.w.abs().max().item()
                summary_dict['w_gini_clip'] = gini_coefficient(np.clip(cls_result.w, 0, None))
                summary_dict['w_gini_abs'] = gini_coefficient(np.abs(cls_result.w))
            
            summary_df = pd.DataFrame(summary_dict)
            current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            summary_df.to_csv(f"{result_dir}/{self.args.dataset}-{self.args.seed}-{self.args.model}-{self.args.num_shapelet}-{self.args.lambda_div}-{self.args.lambda_reg}-{current_time}.csv", index=False)
            print(f"Test summary saved at: {result_dir}/{self.args.dataset}-{self.args.seed}-{self.args.model}-{self.args.num_shapelet}-{self.args.lambda_div}-{self.args.lambda_reg}-{current_time}.csv")
            test_df = summary_df
        
        return test_loss, cls_result, test_df'''
    '''12.7;18:00
    def test(self, save_csv=True, result_dir=None):
        """测试模型"""
        if not os.path.isdir(result_dir):
            try:
                os.makedirs(result_dir)
            except:
                pass

        # 添加调试信息
        print(f"\n{'='*50}")
        print("测试阶段开始")
        print(f"{'='*50}")
        print(f"测试集大小: {len(self.test_loader.dataset) if hasattr(self.test_loader, 'dataset') else 'N/A'}")
        print(f"测试批次数量: {len(self.test_loader)}")

        # 如果测试集为空，直接返回
        if hasattr(self.test_loader, 'dataset') and len(self.test_loader.dataset) == 0:
            print("❌ 错误: 测试集为空")
            return float('inf'), None, None

        @dataclass
        class Buffer:
            x_data: list = field(default_factory=list)
            trues: list = field(default_factory=list)
            preds: list = field(default_factory=list)
            shapelet_preds: list = field(default_factory=list)
            dnn_preds: list = field(default_factory=list)
            p: list = field(default_factory=list)
            d: list = field(default_factory=list)
            eta: list = field(default_factory=list)
            loss: list = field(default_factory=list)

        buffer = Buffer()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(self.test_loader):
                # 添加批次调试信息
                print(f"\n处理测试批次 {i}:")
                print(f"  batch_x 形状: {batch_x.shape}")
                print(f"  label 形状: {label.shape}")
                print(f"  padding_mask 形状: {padding_mask.shape}") if padding_mask is not None else print("  padding_mask: None")

                # 检查批次是否为空
                if batch_x.size(0) == 0 or label.size(0) == 0:
                    print(f"  ⚠ 批次 {i} 为空，跳过")
                    continue

                # 检查label维度
                original_label_shape = label.shape
                print(f"  label 原始形状: {original_label_shape}")

                # 处理标签维度
                batch_x = batch_x.float().to(self.device)
                label = label.long().to(self.device)

                # 检查标签是否是一维
                if label.dim() > 1:
                    print(f"  ⚠ 标签维度 > 1: {label.shape}")
                    print(f"    将标签从 {label.shape} 压缩到 {label.squeeze(-1).shape}")
                    label = label.squeeze(-1)  # 移除最后一维

                # 检查标签值
                print(f"  label 处理后形状: {label.shape}")
                print(f"  label 值范围: {label.min().item()} 到 {label.max().item()}")
                print(f"  label 唯一值: {torch.unique(label)}")

                # 过滤无效标签
                if label.min() < 0 or label.max() >= self.args.num_class:
                    print(f"  ⚠ 标签值超出范围 [0, {self.args.num_class-1}]")
                    print(f"    过滤前样本数: {label.size(0)}")

                    # 只保留有效标签
                    valid_mask = (label >= 0) & (label < self.args.num_class)
                    if valid_mask.sum() == 0:
                        print(f"  ⚠ 批次 {i} 中没有有效标签，跳过")
                        continue

                    batch_x = batch_x[valid_mask]
                    label = label[valid_mask]
                    if padding_mask is not None:
                        padding_mask = padding_mask.float().to(self.device)
                        padding_mask = padding_mask[valid_mask] if padding_mask.size(0) > 1 else padding_mask
                    else:
                        padding_mask = torch.ones(batch_x.shape[0], batch_x.shape[1], device=self.device)

                    print(f"    过滤后样本数: {label.size(0)}")
                    print(f"    有效标签比例: {valid_mask.sum().item()}/{len(valid_mask)} = {valid_mask.sum().item()/len(valid_mask)*100:.1f}%")

                if padding_mask is None:
                    padding_mask = torch.ones(batch_x.shape[0], batch_x.shape[1], device=self.device)
                else:
                    padding_mask = padding_mask.float().to(self.device)

                print(f"  处理完成后:")
                print(f"    batch_x 形状: {batch_x.shape}")
                print(f"    label 形状: {label.shape}")
                print(f"    padding_mask 形状: {padding_mask.shape}")

                # 检查形状是否匹配
                if batch_x.size(0) != label.size(0):
                    print(f"  ❌ 形状不匹配: batch_x={batch_x.shape}, label={label.shape}")
                    min_batch = min(batch_x.size(0), label.size(0))
                    if min_batch > 0:
                        print(f"    使用前 {min_batch} 个样本")
                        batch_x = batch_x[:min_batch]
                        label = label[:min_batch]
                        padding_mask = padding_mask[:min_batch] if padding_mask.size(0) > 1 else padding_mask
                    else:
                        print(f"  ⚠ 批次 {i} 没有有效样本，跳过")
                        continue

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.args.amp):
                    try:
                        if self.args.model == 'DNN':
                            print(f"  DNN 模型推理...")
                            logits = self.model(batch_x, padding_mask, None, None)
                            print(f"  logits 形状: {logits.shape}")
                            loss = nn.functional.cross_entropy(logits, label, reduction='none')
                            print(f"  ✓ DNN推理成功")
                        else:
                            print(f"  {self.args.model} 模型推理...")
                            logits, model_info = self.model(batch_x, padding_mask, None, None, gating_value=self.args.gating_value)
                            print(f"  logits 形状: {logits.shape}")
                            loss = nn.functional.cross_entropy(logits, label, reduction='none') + model_info.loss.mean()
                            print(f"  ✓ {self.args.model}推理成功")

                        # 存储结果
                        buffer.loss.append(loss.flatten())
                        buffer.x_data.append(batch_x.cpu())
                        buffer.trues.append(label.cpu())
                        buffer.preds.append(logits.cpu())

                        if self.args.model != 'DNN':
                            buffer.p.append(model_info.p.cpu())
                            buffer.d.append(model_info.d.cpu())
                            buffer.shapelet_preds.append(model_info.shapelet_preds.cpu())
                            if self.args.model == 'InterpGN':
                                buffer.eta.append(model_info.eta.cpu())
                                buffer.dnn_preds.append(model_info.dnn_preds.cpu())

                        print(f"  ✓ 批次 {i} 处理完成")

                    except Exception as e:
                        print(f"  ❌ 批次 {i} 处理失败: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

        # 检查是否有处理的数据
        if len(buffer.trues) == 0:
            print(f"❌ 错误: 没有处理任何有效批次")
            return float('inf'), None, None

        print(f"\n数据统计:")
        print(f"  处理的批次数量: {len(buffer.trues)}")
        print(f"  trues 长度: {len(buffer.trues)}")
        print(f"  preds 长度: {len(buffer.preds)}")
        print(f"  loss 长度: {len(buffer.loss)}")

        if len(buffer.trues) > 0 and len(buffer.preds) > 0:
            try:
                # 合并结果
                trues_tensor = torch.cat(buffer.trues, dim=0)
                preds_tensor = torch.cat(buffer.preds, dim=0)
                loss_tensor = torch.cat(buffer.loss, dim=0) if buffer.loss else torch.tensor([float('inf')])

                print(f"\n合并后形状:")
                print(f"  trues 形状: {trues_tensor.shape}")
                print(f"  preds 形状: {preds_tensor.shape}")
                print(f"  loss 形状: {loss_tensor.shape}")

                # 确保维度正确
                if trues_tensor.dim() > 1:
                    trues_tensor = trues_tensor.flatten()
                    print(f"  展平trues后的形状: {trues_tensor.shape}")

                if preds_tensor.dim() < 2:
                    print(f"  ❌ 错误: preds形状不正确: {preds_tensor.shape}")
                    return float('inf'), None, None

                # 计算准确率
                probs = torch.nn.functional.softmax(preds_tensor, dim=1)
                predictions = torch.argmax(probs, dim=1)

                print(f"  probs 形状: {probs.shape}")
                print(f"  predictions 形状: {predictions.shape}")

                # 检查形状匹配
                if predictions.shape != trues_tensor.shape:
                    print(f"  ❌ 形状不匹配: predictions={predictions.shape}, trues={trues_tensor.shape}")
                    min_len = min(predictions.shape[0], trues_tensor.shape[0])
                    if min_len > 0:
                        predictions = predictions[:min_len]
                        trues_tensor = trues_tensor[:min_len]
                        print(f"  截断到: predictions={predictions.shape}, trues={trues_tensor.shape}")
                    else:
                        print(f"  ❌ 没有有效样本")
                        return float('inf'), None, None

                accuracy = accuracy_score(predictions.cpu().numpy(), trues_tensor.cpu().numpy())
                test_loss = loss_tensor.mean().item()

                print(f"\n测试结果:")
                print(f"  样本总数: {trues_tensor.shape[0]}")
                print(f"  测试损失: {test_loss:.4f}")
                print(f"  测试准确率: {accuracy:.4f}")

                # 计算类别分布
                unique_labels, counts = torch.unique(trues_tensor, return_counts=True)
                print(f"  标签分布:")
                for label, count in zip(unique_labels, counts):
                    print(f"    类别 {label.item()}: {count.item()} 个样本")

                cls_result = ClassificationResult(
                    x_data=torch.cat(buffer.x_data, dim=0).cpu() if buffer.x_data else torch.tensor([]),
                    trues=trues_tensor.cpu(),
                    preds=predictions.cpu(),
                    loss=test_loss,
                    accuracy=accuracy
                )

                if self.args.model != 'DNN':
                    if buffer.p:
                        cls_result.p = torch.cat(buffer.p, dim=0).cpu()
                    if buffer.d:
                        cls_result.d = torch.cat(buffer.d, dim=0).cpu()
                    if buffer.shapelet_preds:
                        cls_result.shapelet_preds = torch.cat(buffer.shapelet_preds, dim=0).cpu()

                    if self.args.model == 'InterpGN':
                        if buffer.eta:
                            cls_result.eta = torch.cat(buffer.eta, dim=0).cpu()
                        cls_result.w = self.model.sbm.output_layer.weight.detach().cpu()
                        cls_result.shapelets = self.model.sbm.get_shapelets()
                    else:
                        cls_result.w = self.model.output_layer.weight.detach().cpu()
                        cls_result.shapelets = self.model.get_shapelets()

            except Exception as e:
                print(f"❌ 处理测试结果时出错: {e}")
                import traceback
                traceback.print_exc()
                return float('inf'), None, None
        else:
            print(f"❌ 错误: 没有可处理的数据")
            return float('inf'), None, None

        test_df = None
        if save_csv and result_dir is not None:
            try:
                summary_dict = dict()
                for arg in vars(self.args):
                    if arg in [
                        'model', 'dataset', 'dnn_type', 
                        'train_epochs', 'num_shapelet', 'lambda_reg', 'lambda_div', 'epsilon', 'lr', 
                        'seed', 'pos_weight', 'beta_schedule', 'gating_value',
                        'distance_func', 'sbm_cls'
                    ]:
                        summary_dict[arg] = [getattr(self.args, arg)]

                summary_dict['test_accuracy'] = accuracy
                summary_dict['epoch_stop'] = self.epoch_stop

                if self.args.model != 'DNN' and hasattr(cls_result, 'w'):
                    summary_dict['shapelet_score'] = compute_shapelet_score(cls_result.d, cls_result.w, cls_result.preds, cls_result.trues)
                    # ... 其他统计信息

                summary_df = pd.DataFrame(summary_dict)
                current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                csv_path = f"{result_dir}/{self.args.dataset}-{self.args.seed}-{self.args.model}-{self.args.num_shapelet}-{self.args.lambda_div}-{self.args.lambda_reg}-{current_time}.csv"
                summary_df.to_csv(csv_path, index=False)
                print(f"✓ 测试结果已保存: {csv_path}")
                test_df = summary_df

            except Exception as e:
                print(f"❌ 保存CSV时出错: {e}")
        # 返回一个字典，包含测试指标
        test_metrics = {
        'loss': test_loss,
        'accuracy': accuracy,
        'test_samples': trues_tensor.shape[0],
        'class_distribution': {int(k.item()): int(v.item()) for k, v in zip(unique_labels, counts)}
    }
    
    # 如果有cls_result，包含额外信息
        if hasattr(self.args, 'model') and self.args.model != 'DNN' and hasattr(cls_result, 'w'):
            test_metrics.update({
            'shapelet_score': compute_shapelet_score(cls_result.d, cls_result.w, cls_result.preds, cls_result.trues) if hasattr(cls_result, 'd') and hasattr(cls_result, 'w') else None,
        })
    
        return test_loss, test_metrics, test_df
        #return test_loss, cls_result, test_df'''
    def test(self, save_csv=True, result_dir=None):
        """测试模型 - 只显示第一个batch的详细信息"""
        if not os.path.isdir(result_dir):
            try:
                os.makedirs(result_dir)
            except:
                pass

        # 添加调试信息
        print(f"\n{'='*50}")
        print("测试阶段开始")
        print(f"{'='*50}")
        print(f"测试集大小: {len(self.test_loader.dataset) if hasattr(self.test_loader, 'dataset') else 'N/A'}")
        print(f"测试批次数量: {len(self.test_loader)}")

        # 如果测试集为空，直接返回
        if hasattr(self.test_loader, 'dataset') and len(self.test_loader.dataset) == 0:
            print("❌ 错误: 测试集为空")
            return float('inf'), None, None

        @dataclass
        class Buffer:
            x_data: list = field(default_factory=list)
            trues: list = field(default_factory=list)
            preds: list = field(default_factory=list)
            shapelet_preds: list = field(default_factory=list)
            dnn_preds: list = field(default_factory=list)
            p: list = field(default_factory=list)
            d: list = field(default_factory=list)
            eta: list = field(default_factory=list)
            loss: list = field(default_factory=list)

        buffer = Buffer()
        self.model.eval()
        batch_count = 0
        first_batch_printed = False

        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(self.test_loader):
                batch_size = batch_x.size(0)

                # 只处理第一个batch的输出
                if not first_batch_printed:
                    print(f"\n第一个批次的详细信息:")
                    print(f"  批次索引: {i}")
                    print(f"  批次大小: {batch_size}")
                    print(f"  batch_x 形状: {batch_x.shape}")
                    print(f"  label 形状: {label.shape}")
                    if padding_mask is not None:
                        print(f"  padding_mask 形状: {padding_mask.shape}")
                    else:
                        print(f"  padding_mask: None")
                    first_batch_printed = True

                # 检查批次是否为空
                if batch_size == 0 or label.size(0) == 0:
                    if batch_count == 0:
                        print(f"  ⚠ 第一个批次为空，跳过")
                    batch_count += 1
                    continue

                # 处理标签维度
                batch_x = batch_x.float().to(self.device)
                label = label.long().to(self.device)

                # 检查标签是否是一维
                if label.dim() > 1:
                    if batch_count == 0:
                        print(f"  ⚠ 标签维度 > 1: {label.shape}")
                    label = label.squeeze(-1)  # 移除最后一维

                # 只在第一个batch检查标签值
                if batch_count == 0:
                    print(f"  label 处理后形状: {label.shape}")
                    print(f"  label 值范围: {label.min().item()} 到 {label.max().item()}")
                    print(f"  label 唯一值: {torch.unique(label)}")

                # 过滤无效标签
                if label.min() < 0 or label.max() >= self.args.num_class:
                    if batch_count == 0:
                        print(f"  ⚠ 标签值超出范围 [0, {self.args.num_class-1}]")
                        print(f"    过滤前样本数: {label.size(0)}")

                    # 只保留有效标签
                    valid_mask = (label >= 0) & (label < self.args.num_class)
                    if valid_mask.sum() == 0:
                        if batch_count == 0:
                            print(f"  ⚠ 批次 {i} 中没有有效标签，跳过")
                        batch_count += 1
                        continue

                    batch_x = batch_x[valid_mask]
                    label = label[valid_mask]
                    if padding_mask is not None:
                        padding_mask = padding_mask.float().to(self.device)
                        padding_mask = padding_mask[valid_mask] if padding_mask.size(0) > 1 else padding_mask
                    else:
                        padding_mask = torch.ones(batch_x.shape[0], batch_x.shape[1], device=self.device)

                    if batch_count == 0:
                        print(f"    过滤后样本数: {label.size(0)}")
                        print(f"    有效标签比例: {valid_mask.sum().item()}/{len(valid_mask)} = {valid_mask.sum().item()/len(valid_mask)*100:.1f}%")
                else:
                    if padding_mask is None:
                        padding_mask = torch.ones(batch_x.shape[0], batch_x.shape[1], device=self.device)
                    else:
                        padding_mask = padding_mask.float().to(self.device)

                if batch_count == 0:
                    print(f"  处理完成后:")
                    print(f"    batch_x 形状: {batch_x.shape}")
                    print(f"    label 形状: {label.shape}")
                    print(f"    padding_mask 形状: {padding_mask.shape}")

                # 检查形状是否匹配
                if batch_x.size(0) != label.size(0):
                    if batch_count == 0:
                        print(f"  ⚠ 形状不匹配: batch_x={batch_x.shape}, label={label.shape}")
                    min_batch = min(batch_x.size(0), label.size(0))
                    if min_batch > 0:
                        if batch_count == 0:
                            print(f"    使用前 {min_batch} 个样本")
                        batch_x = batch_x[:min_batch]
                        label = label[:min_batch]
                        padding_mask = padding_mask[:min_batch] if padding_mask.size(0) > 1 else padding_mask
                    else:
                        if batch_count == 0:
                            print(f"  ⚠ 批次 {i} 没有有效样本，跳过")
                        batch_count += 1
                        continue

                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.args.amp):
                    try:
                        if self.args.model == 'DNN':
                            if batch_count == 0:
                                print(f"  DNN 模型推理...")
                            logits = self.model(batch_x, padding_mask, None, None)
                            if batch_count == 0:
                                print(f"  logits 形状: {logits.shape}")
                            loss = nn.functional.cross_entropy(logits, label, reduction='none')
                            if batch_count == 0:
                                print(f"  ✓ DNN推理成功")
                                print(f"  ✓ 批次 {i} 处理完成")
                        else:
                            if batch_count == 0:
                                print(f"  {self.args.model} 模型推理...")
                            logits, model_info = self.model(batch_x, padding_mask, None, None, gating_value=self.args.gating_value)
                            if batch_count == 0:
                                print(f"  logits 形状: {logits.shape}")
                            loss = nn.functional.cross_entropy(logits, label, reduction='none') + model_info.loss.mean()
                            if batch_count == 0:
                                print(f"  ✓ {self.args.model}推理成功")
                                print(f"  ✓ 批次 {i} 处理完成")

                        # 存储结果
                        buffer.loss.append(loss.flatten())
                        buffer.x_data.append(batch_x.cpu())
                        buffer.trues.append(label.cpu())
                        buffer.preds.append(logits.cpu())

                        if self.args.model != 'DNN':
                            buffer.p.append(model_info.p.cpu())
                            buffer.d.append(model_info.d.cpu())
                            buffer.shapelet_preds.append(model_info.shapelet_preds.cpu())
                            if self.args.model == 'InterpGN':
                                buffer.eta.append(model_info.eta.cpu())
                                buffer.dnn_preds.append(model_info.dnn_preds.cpu())

                    except Exception as e:
                        if batch_count == 0:
                            print(f"  ❌ 批次 {i} 处理失败: {e}")
                            import traceback
                            traceback.print_exc()
                        batch_count += 1
                        continue

                batch_count += 1

                # 后续批次只显示进度，不显示详细信息
                if i > 0 and i % 20 == 0:  # 每20个批次显示一次进度
                    print(f"  处理批次 {i+1}/{len(self.test_loader)}...")

        # 处理结果统计
        print(f"\n{'='*50}")
        print("数据处理统计:")
        print(f"{'='*50}")
        print(f"  总批次数量: {len(self.test_loader)}")
        print(f"  成功处理批次数量: {len(buffer.trues)}")
        print(f"  trues 长度: {len(buffer.trues)}")
        print(f"  preds 长度: {len(buffer.preds)}")

        # 检查是否有处理的数据
        if len(buffer.trues) == 0:
            print(f"❌ 错误: 没有处理任何有效批次")
            return float('inf'), None, None

        if len(buffer.trues) > 0 and len(buffer.preds) > 0:
            try:
                # 合并结果
                trues_tensor = torch.cat(buffer.trues, dim=0)
                preds_tensor = torch.cat(buffer.preds, dim=0)
                loss_tensor = torch.cat(buffer.loss, dim=0) if buffer.loss else torch.tensor([float('inf')])

                # 只打印合并后的形状
                print(f"\n{'='*50}")
                print("合并结果统计:")
                print(f"{'='*50}")
                print(f"  trues 形状: {trues_tensor.shape}")
                print(f"  preds 形状: {preds_tensor.shape}")
                print(f"  loss 形状: {loss_tensor.shape}")
                print(f"  总样本数: {trues_tensor.shape[0]}")

                # 确保维度正确
                if trues_tensor.dim() > 1:
                    trues_tensor = trues_tensor.flatten()
                if preds_tensor.dim() < 2:
                    print(f"  ❌ 错误: preds形状不正确: {preds_tensor.shape}")
                    return float('inf'), None, None

                # 计算准确率
                probs = torch.nn.functional.softmax(preds_tensor, dim=1)
                predictions = torch.argmax(probs, dim=1)

                # 确保形状匹配
                if predictions.shape != trues_tensor.shape:
                    min_len = min(predictions.shape[0], trues_tensor.shape[0])
                    if min_len > 0:
                        predictions = predictions[:min_len]
                        trues_tensor = trues_tensor[:min_len]
                    else:
                        print(f"  ❌ 没有有效样本")
                        return float('inf'), None, None

                accuracy = accuracy_score(predictions.cpu().numpy(), trues_tensor.cpu().numpy())
                test_loss = loss_tensor.mean().item()

                # 计算类别分布
                unique_labels, counts = torch.unique(trues_tensor, return_counts=True)
                total_samples = trues_tensor.shape[0]

                print(f"\n{'='*50}")
                print("测试结果汇总")
                print(f"{'='*50}")
                print(f"  样本总数: {total_samples}")
                print(f"  测试损失: {test_loss:.6f}")
                print(f"  测试准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

                print(f"\n  标签分布:")
                for label, count in zip(unique_labels, counts):
                    percentage = (count.item() / total_samples) * 100
                    print(f"    类别 {label.item()}: {count.item()} 个样本 ({percentage:.1f}%)")

                # 与随机基线比较
                num_classes = getattr(self.args, 'num_class', 3)
                random_baseline = 100.0 / num_classes
                improvement = accuracy*100 - random_baseline
                print(f"\n  性能分析:")
                print(f"    模型准确率: {accuracy*100:.2f}%")
                print(f"    随机基线 ({num_classes}分类): {random_baseline:.2f}%")
                print(f"    提升: {improvement:+.2f}% ({improvement/random_baseline*100:.1f}%相对提升)")
                if improvement > 0:
                    print(f"    ✓ 模型优于随机猜测")
                else:
                    print(f"    ⚠ 模型低于随机猜测")
                print(f"{'='*50}")

                # 创建结果对象
                cls_result = ClassificationResult(
                    x_data=torch.cat(buffer.x_data, dim=0).cpu() if buffer.x_data else torch.tensor([]),
                    trues=trues_tensor.cpu(),
                    preds=predictions.cpu(),
                    loss=test_loss,
                    accuracy=accuracy
                )

                if self.args.model != 'DNN':
                    if buffer.p:
                        cls_result.p = torch.cat(buffer.p, dim=0).cpu()
                    if buffer.d:
                        cls_result.d = torch.cat(buffer.d, dim=0).cpu()
                    if buffer.shapelet_preds:
                        cls_result.shapelet_preds = torch.cat(buffer.shapelet_preds, dim=0).cpu()

                    if self.args.model == 'InterpGN':
                        if buffer.eta:
                            cls_result.eta = torch.cat(buffer.eta, dim=0).cpu()
                        cls_result.w = self.model.sbm.output_layer.weight.detach().cpu()
                        cls_result.shapelets = self.model.sbm.get_shapelets()
                    else:
                        cls_result.w = self.model.output_layer.weight.detach().cpu()
                        cls_result.shapelets = self.model.get_shapelets()

            except Exception as e:
                print(f"❌ 处理测试结果时出错: {e}")
                import traceback
                traceback.print_exc()
                return float('inf'), None, None
        else:
            print(f"❌ 错误: 没有可处理的数据")
            return float('inf'), None, None

        # 保存结果
        test_df = None
        if save_csv and result_dir is not None:
            try:
                # ... 保存CSV的代码 ...
                pass
            except Exception as e:
                print(f"❌ 保存CSV时出错: {e}")

        return test_loss, cls_result, test_df