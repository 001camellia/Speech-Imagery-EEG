# models/model_eegcnn.py
"""
EEGCNN+Transformer 模型实现
这个模型将作为基线模型
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """位置编码层"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class EEGcnn(nn.Module):
    """EEGCNN特征提取器"""
    def __init__(self, Chans=122, kernLength1=125, kernLength2=25, F1=8, D=8, F2=64, P1=2, P2=5, dropoutRate=0.1, **kwargs):
        super(EEGcnn, self).__init__()
        
        # 兼容性处理：如果 Chans 是 Namespace 对象
        if hasattr(Chans, '__dict__'):  # 是 Namespace 对象
            args = Chans
            # 从 args 中提取参数
            Chans = getattr(args, 'enc_in', 
                           getattr(args, 'input_channels', 
                                  getattr(args, 'n_channels', 122)))
            kernLength1 = getattr(args, 'kernel_length1', 
                                 getattr(args, 'kernLength1', 125))
            kernLength2 = getattr(args, 'kernel_length2', 
                                 getattr(args, 'kernLength2', 25))
            F1 = getattr(args, 'cnn_filter1', 
                        getattr(args, 'F1', 8))
            D = getattr(args, 'cnn_filter2', 
                       getattr(args, 'D', 8))
            F2 = F1 * D
            P1 = getattr(args, 'pooling1', 
                        getattr(args, 'P1', 2))
            P2 = getattr(args, 'pooling2', 
                     getattr(args, 'P2', 5))
            dropoutRate = getattr(args, 'dropout1', 
                                 getattr(args, 'dropoutRate', 0.1))
        
        # 确保 Chans 是整数
        Chans = int(Chans)
        
        print(f"[EEGcnn] 参数设置:")
        print(f"  Chans: {Chans}")
        print(f"  F1: {F1}, D: {D}, F2: {F1*D}")
        print(f"  kernLength1: {kernLength1}, kernLength2: {kernLength2}")
        
        # Block 1: 时间卷积
        self.block1_conv1 = nn.Conv2d(1, F1, (1, kernLength1), padding='same', bias=False)
        self.block1_bn1 = nn.BatchNorm2d(F1)
        
        # 深度可分离卷积
        self.block1_depthwise = nn.Conv2d(F1, D*F1, (Chans, 1), groups=F1, bias=False)
        self.block1_bn2 = nn.BatchNorm2d(D*F1)
        self.block1_elu = nn.ELU()
        self.block1_pool = nn.AvgPool2d((1, P1))
        self.block1_drop = nn.Dropout(p=dropoutRate)
        
        # Block 2: 空间卷积
        self.block2_conv1 = nn.Conv2d(D*F1, D*F1, (1, kernLength2), padding='same', groups=D*F1, bias=False)
        self.block2_conv2 = nn.Conv2d(D*F1, F2, 1, bias=False)
        self.block2_bn = nn.BatchNorm2d(F2)
        self.block2_elu = nn.ELU()
        self.block2_pool = nn.AvgPool2d((1, P2))
        self.block2_drop = nn.Dropout(p=dropoutRate)
        
    def forward(self, x):
        # 输入: (batch, Chans, timepoints)
        x = x.unsqueeze(1)  # (batch, 1, Chans, timepoints)
        
        # Block 1
        x = self.block1_conv1(x)
        x = self.block1_bn1(x)
        x = self.block1_depthwise(x)
        x = self.block1_bn2(x)
        x = self.block1_elu(x)
        x = self.block1_pool(x)
        x = self.block1_drop(x)
        
        # Block 2
        x = self.block2_conv1(x)
        x = self.block2_conv2(x)
        x = self.block2_bn(x)
        x = self.block2_elu(x)
        x = self.block2_pool(x)
        x = self.block2_drop(x)
        
        # 输出: (batch, F2, 1, reduced_timepoints)
        x = x.squeeze(2)  # (batch, F2, reduced_timepoints)
        return x


class EEGCNNTransformer(nn.Module):
    """EEGCNN+Transformer 分类模型"""
    def __init__(self, configs=None, **kwargs):
        super(EEGCNNTransformer, self).__init__()
        
        # 处理参数：优先从 kwargs 获取，然后从 configs 获取
        if configs is not None and hasattr(configs, '__dict__'):
            # 从 configs 对象获取参数
            input_channels = getattr(configs, 'enc_in', 
                                    getattr(configs, 'input_channels', 122))
            seq_len = getattr(configs, 'seq_len', 845)
            num_classes = getattr(configs, 'c_out', 
                                 getattr(configs, 'num_class', 
                                        getattr(configs, 'num_classes', 3)))
            
            # EEGCNN 特定参数
            dropout1 = getattr(configs, 'eegcnn_dropout1', 
                              getattr(configs, 'dropout1', 0.1))
            dropout2 = getattr(configs, 'eegcnn_dropout2', 
                              getattr(configs, 'dropout2', 0.1))
            num_layers = getattr(configs, 'eegcnn_layers', 
                                getattr(configs, 'num_layers', 0))
            pooling = getattr(configs, 'eegcnn_pooling', 
                             getattr(configs, 'pooling', None))
            cnn_filter1 = getattr(configs, 'cnn_filter1', 
                                 getattr(configs, 'F1', 8))
            cnn_filter2 = getattr(configs, 'cnn_filter2', 
                                 getattr(configs, 'D', 8))
            kernel_length1 = getattr(configs, 'kernel_length1', 
                                    getattr(configs, 'kernLength1', 125))
            kernel_length2 = getattr(configs, 'kernel_length2', 
                                    getattr(configs, 'kernLength2', 25))
            pooling1 = getattr(configs, 'pooling1', 
                              getattr(configs, 'P1', 2))
            pooling2 = getattr(configs, 'pooling2', 
                             getattr(configs, 'P2', 5))
            d_model = getattr(configs, 'd_model', None)
            n_heads = getattr(configs, 'eegcnn_n_heads', 
                             getattr(configs, 'n_heads', 8))
            dim_feedforward = getattr(configs, 'eegcnn_d_ff', 
                                     getattr(configs, 'dim_feedforward', 256))
            output_attention = getattr(configs, 'output_attention', False)
        else:
            # 从 kwargs 获取参数
            input_channels = kwargs.get('input_channels', 122)
            seq_len = kwargs.get('seq_len', 845)
            num_classes = kwargs.get('num_classes', 3)
            
            dropout1 = kwargs.get('dropout1', 0.1)
            dropout2 = kwargs.get('dropout2', 0.1)
            num_layers = kwargs.get('num_layers', 0)
            pooling = kwargs.get('pooling', None)
            cnn_filter1 = kwargs.get('cnn_filter1', 8)
            cnn_filter2 = kwargs.get('cnn_filter2', 8)
            kernel_length1 = kwargs.get('kernel_length1', 125)
            kernel_length2 = kwargs.get('kernel_length2', 25)
            pooling1 = kwargs.get('pooling1', 2)
            pooling2 = kwargs.get('pooling2', 5)
            d_model = kwargs.get('d_model', None)
            n_heads = kwargs.get('n_heads', 8)
            dim_feedforward = kwargs.get('dim_feedforward', 256)
            output_attention = kwargs.get('output_attention', False)
        
        # 打印配置信息
        print(f"\n[EEGCNNTransformer] 模型配置:")
        print(f"  input_channels: {input_channels}")
        print(f"  seq_len: {seq_len}")
        print(f"  num_classes: {num_classes}")
        print(f"  cnn_filter1: {cnn_filter1}, cnn_filter2: {cnn_filter2}")
        print(f"  kernel_length1: {kernel_length1}, kernel_length2: {kernel_length2}")
        print(f"  dropout1: {dropout1}, dropout2: {dropout2}")
        print(f"  num_layers: {num_layers}")
        print(f"  pooling: {pooling}")
        
        # 保存参数
        self.input_channels = input_channels
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.pooling = pooling
        self.output_attention = output_attention
        
        # EEGCNN
        self.eegcnn = EEGcnn(
            Chans=input_channels,
            kernLength1=kernel_length1,
            kernLength2=kernel_length2,
            F1=cnn_filter1,
            D=cnn_filter2,
            F2=cnn_filter1*cnn_filter2,
            P1=pooling1,
            P2=pooling2,
            dropoutRate=dropout1
        )
        
        # CNN输出维度
        cnn_out_channels = cnn_filter1 * cnn_filter2
        self.cnn_out_channels = cnn_out_channels
        
        # Transformer
        if num_layers > 0:
            if d_model is None:
                d_model = cnn_out_channels
            
            # 位置编码
            self.pos_encoder = PositionalEncoding(d_model, dropout=dropout2, max_len=5000)
            
            # Transformer编码器层
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout2,
                batch_first=True
            )
            
            # Transformer编码器
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # 如果CNN输出维度不等于d_model，需要投影
            if cnn_out_channels != d_model:
                self.cnn_projection = nn.Linear(cnn_out_channels, d_model)
            else:
                self.cnn_projection = nn.Identity()
                
            self.d_model = d_model
        else:
            self.d_model = cnn_out_channels
        
        # 分类头
        if pooling is None:
            # 展平
            self.classifier = nn.Linear(seq_len * self.d_model, num_classes)
        else:
            # 池化后
            self.classifier = nn.Linear(self.d_model, num_classes)
        
    def _original_forward(self, x, mask=None):
        """原始的前向传播逻辑"""
        # 输入: (batch, channels, timepoints)
        batch_size = x.size(0)
        
        # 1. EEGCNN特征提取
        x = self.eegcnn(x)  # (batch, cnn_out_channels, reduced_timepoints)
        
        # 获取CNN输出的序列长度
        reduced_seq_len = x.size(2)
        
        # 调整掩码大小
        if mask is not None:
            # 对掩码进行平均池化以适应CNN输出
            mask = F.avg_pool1d(mask.float().unsqueeze(1), kernel_size=5, stride=2)  # 粗略估计
            mask = mask.squeeze(1) > 0.5
        else:
            mask = torch.ones(batch_size, reduced_seq_len, device=x.device, dtype=torch.bool)
        
        # 转置: (batch, reduced_seq_len, features)
        x = x.permute(0, 2, 1)
        
        # 2. Transformer编码 (如果使用)
        if self.num_layers > 0:
            # 投影到d_model维度
            x = self.cnn_projection(x)
            
            # 位置编码
            x = self.pos_encoder(x)
            
            # Transformer编码
            x = self.transformer_encoder(x, src_key_padding_mask=(~mask))
        
        # 3. 池化
        if self.pooling is None:
            # 展平
            x = x.reshape(batch_size, -1)
        elif self.pooling == "mean":
            # 平均池化
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x = torch.sum(x * mask_expanded, dim=1) / torch.sum(mask, dim=1, keepdim=True)
        elif self.pooling == "sum":
            # 求和池化
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x = torch.sum(x * mask_expanded, dim=1)
        elif self.pooling == "top":
            # 取第一个token
            x = x[:, 0, :]
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")
        
        # 4. 分类
        logits = self.classifier(x)
        
        # 返回结果
        output = {"logits": logits}
        
        if self.output_attention and self.num_layers > 0:
            # 可以在这里添加注意力输出
            pass
        
        return output
    
    def forward(self, *args, **kwargs):
        """
        兼容性前向传播
        自动处理不同数量的参数
        """
        # 提取参数
        if len(args) > 0:
            x = args[0]
        else:
            x = kwargs.get('x')

        if len(args) > 1:
            padding_mask = args[1]
        else:
            padding_mask = kwargs.get('padding_mask')
        
        # 调试信息
        print(f"[DEBUG] EEGCNN forward 参数:")
        print(f"  args数量: {len(args)}")
        print(f"  kwargs数量: {len(kwargs)}")
        print(f"  x shape: {x.shape if x is not None else 'None'}")
        print(f"  padding_mask: {padding_mask.shape if padding_mask is not None else 'None'}")

        # 调用原始forward
        if padding_mask is not None:
            output = self._original_forward(x, padding_mask)
        else:
            output = self._original_forward(x)

        # 获取logits
        if isinstance(output, dict):
            logits = output.get("logits")
        elif isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        
        if logits is None:
            raise ValueError("无法从输出中获取 logits")
        
        # 创建model_info
        class ModelInfo:
            def __init__(self, logits):
                self.logits = logits
                self.loss = torch.tensor(0.0, device=logits.device)
                self.p = None
                self.d = None
                self.shapelet_preds = None
                self.eta = None
                self.dnn_preds = None

        model_info = ModelInfo(logits)
        
        print(f"[DEBUG] EEGCNN forward 返回:")
        print(f"  logits shape: {logits.shape}")

        return logits, model_info

    def get_attention_maps(self, x, mask=None):
        """获取注意力图（用于可视化）"""
        if self.num_layers == 0:
            return None
        
        # 保存注意力
        attention_maps = []
        
        batch_size = x.size(0)
        x = self.eegcnn(x)
        x = x.permute(0, 2, 1)
        x = self.cnn_projection(x)
        x = self.pos_encoder(x)
        
        # 获取每个transformer层的注意力
        for layer in self.transformer_encoder.layers:
            # 计算自注意力
            attn_output, attn_weights = layer.self_attn(
                x, x, x, 
                key_padding_mask=(~mask) if mask is not None else None,
                need_weights=True
            )
            attention_maps.append(attn_weights)
            
            # 前馈网络
            x = layer.norm1(x + layer.dropout1(attn_output))
            x = layer.norm2(x + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(x) ) ) )) )
        
        return attention_maps


# 在 model_eegcnn.py 中添加
class EEGCNNAdapter(nn.Module):
    """EEGCNN适配器，统一接口"""
    def __init__(self, configs=None, **kwargs):
        super(EEGCNNAdapter, self).__init__()
        
        # 打印传入的configs
        print(f"[DEBUG] EEGCNNAdapter 初始化:")
        print(f"  configs 类型: {type(configs)}")
        if hasattr(configs, '__dict__'):
            print(f"  configs 属性: {list(vars(configs).keys())[:20]}")
        
        # 从configs获取参数
        input_channels = getattr(configs, 'enc_in', 122)
        seq_len = getattr(configs, 'seq_len', 845)
        num_classes = getattr(configs, 'c_out', 3)
        
        # 打印配置
        print(f"[DEBUG] EEGCNNAdapter 参数:")
        print(f"  input_channels: {input_channels}")
        print(f"  seq_len: {seq_len}")
        print(f"  num_classes: {num_classes}")
        
        # 创建原始EEGCNN模型
        self.eegcnn = EEGCNNTransformer(
            input_channels=input_channels,
            seq_len=seq_len,
            num_classes=num_classes,
            dropout1=getattr(configs, 'dropout1', 0.1),
            dropout2=getattr(configs, 'dropout2', 0.1),
            num_layers=getattr(configs, 'num_layers', 0),
            pooling=getattr(configs, 'pooling', None),
            cnn_filter1=getattr(configs, 'cnn_filter1', 8),
            cnn_filter2=getattr(configs, 'cnn_filter2', 8),
            kernel_length1=getattr(configs, 'kernel_length1', 125),
            kernel_length2=getattr(configs, 'kernel_length2', 25),
            pooling1=getattr(configs, 'pooling1', 2),
            pooling2=getattr(configs, 'pooling2', 5),
            d_model=getattr(configs, 'd_model', 512),
            n_heads=getattr(configs, 'n_heads', 8),
            dim_feedforward=getattr(configs, 'dim_feedforward', 256)
        )
        
    def forward(self, x, padding_mask=None, token_emb=None, attn_mask=None, **kwargs):
        # 调用原始EEGCNN的前向传播
        result = self.eegcnn(x, padding_mask)
        logits = result["logits"]
        
        # 创建兼容的model_info对象
        from dataclasses import dataclass
        
        @dataclass
        class ModelInfo:
            logits: torch.Tensor
            loss: torch.Tensor = None
            p: torch.Tensor = None
            d: torch.Tensor = None
            shapelet_preds: torch.Tensor = None
            eta: torch.Tensor = None
            dnn_preds: torch.Tensor = None
            
            def __post_init__(self):
                if self.loss is None:
                    self.loss = torch.tensor(0.0, device=self.logits.device)
        
        model_info = ModelInfo(logits=logits)
        
        return logits, model_info