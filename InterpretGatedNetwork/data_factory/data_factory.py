from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, Monashloader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

# 在data_loader.py中添加
from .eeg import EEGDataset, EEGDataset3Class, eeg_collate_fn

# 更新data_dict
data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'Monash': Monashloader,
    'EEG': EEGDataset,           # 39分类
    'EEG3': EEGDataset3Class,    # 3分类
}


def data_provider(args, flag, bin_edges=None):
    #大小写对应
    # 转换flag为小写
    flag = flag.lower()
    
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    
    
    
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        
        # 为EEG数据使用专门的collate_fn
        if args.data in ['EEG', 'EEG3']:
            # EEG数据特殊处理
            collate_fn_to_use = eeg_collate_fn
            
            # === 简化: 只传递必要的参数，移除target_fs和target_channels ===
            data_set = Data(
                root_path=args.root_path,
                flag=flag,
                json_path=args.json_path,
                max_files=args.max_files,
                debug=getattr(args, 'debug', False),
                test_size=getattr(args, 'test_size', 0.2),
                val_size=getattr(args, 'val_size', 0.1),
                size=[args.seq_len, args.label_len, args.pred_len] if hasattr(args, 'seq_len') else None
                # 不再传递 target_fs 和 target_channels
            )
        else:
            # 其他数据集
            collate_fn_to_use = lambda x: collate_fn(x, max_len=args.seq_len)
            data_set = Data(
                root_path=args.root_path,
                flag=flag,
            )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=collate_fn_to_use
        )
        return data_set, data_loader
      
    elif args.task_name == 'regression':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
            bin_edges=bin_edges
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
