from data_provider.data_loader import Dataset_Glucose
from torch.utils.data import DataLoader

data_dict = {
    'Glucose': Dataset_Glucose,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
        data_path = args.data_path  # Use the original data path for testing in training phase
    elif flag == 'new_test':  # Configuration for the new test dataset
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
        data_path = args.test_data_path  # Use the separate test dataset path
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
        data_path = args.data_path  # Use the original data path for train and val

    data_set = Data(
        root_path=args.root_path,
        data_path=data_path,  # Use the data_path determined above
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        percent=percent,
        seasonal_patterns=args.seasonal_patterns
    )

    if hasattr(data_set, 'feature_names'):
        args.feature_names = data_set.feature_names

    if hasattr(data_set, 'time_features_count'):
        args.time_features_count = data_set.time_features_count

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
