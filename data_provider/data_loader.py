import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Glucose(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='glucose.csv',
                 target='glucose', scale=True, timeenc=0, freq='t', percent=100,
                 seasonal_patterns=None):
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val', 'new_test']
        type_map = {'train':0, 'val':1, 'test':2, 'new_test':3}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = None
        if self.set_type != 3:
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
            
            # Dynamically get all additional features
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('ts')
            self.additional_features = cols
            
            # Reorder columns with ts and target first, followed by additional features
            df_raw = df_raw[['ts', self.target] + self.additional_features]
            
            # Convert ts column to datetime
            df_raw['ts'] = pd.to_datetime(df_raw['ts'])
            
            # Handle missing values based on column type
            for col in self.additional_features:
                # Determine if column contains string data
                is_string_column = df_raw[col].dtype == 'object'
                if is_string_column:
                    df_raw[col] = df_raw[col].fillna('')  # Fill string columns with empty string
                else:
                    df_raw[col] = df_raw[col].fillna(-1)  # Fill numeric columns with -1

            if self.percent != 100:
                data_len = len(df_raw)
                num_samples = int(data_len * self.percent / 100)
                df_raw = df_raw[:num_samples]

            # Explicitly split the dataset based on flag
            train_ratio = 0.7
            valid_ratio = 0.2
            train_len = int(len(df_raw) * train_ratio)
            valid_len = int(len(df_raw) * valid_ratio)

            if self.set_type == 0:
                border1 = 0
                border2 = train_len
            elif self.set_type == 1:
                border1 = train_len
                border2 = train_len + valid_len
            elif self.set_type == 2:
                border1 = train_len + valid_len
                border2 = len(df_raw)
        else: # new_test
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
            
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('ts')
            self.additional_features = cols
            
            df_raw = df_raw[['ts', self.target] + self.additional_features]
            
            # convert the column 'ts' to datetime type
            df_raw['ts'] = pd.to_datetime(df_raw['ts'])
            # deal with missing values
            df_raw = df_raw.fillna(method='ffill')  # Forward fill missing values

            border1 = 0
            border2 = len(df_raw)

        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(df_data[border1:border2].values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Get time features
        df_stamp = df_raw[['ts']][border1:border2]
        df_stamp['ts'] = pd.to_datetime(df_stamp['ts'])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        # Record the number of time features
        self.time_features_count = data_stamp.shape[1]  # This will be the number of time features
        
        # Add additional features
        for feature in self.additional_features:
            feature_data = df_raw[feature][border1:border2].values
            data_stamp = np.concatenate([data_stamp, feature_data.reshape(-1, 1)], axis=1)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        all_columns = df_raw.columns.tolist()
        self.feature_names = [col for col in all_columns 
                            if col not in ['ts', self.target]]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def time_features(dates, timeenc=1, freq='h'):  # Define time_features here
    """
    > `time_features` takes in a `dates` dataframe with a 'ts' column and extracts time features from it.

    Args:
      dates (`pd.DataFrame`): pd.DataFrame with a 'ts' column
      timeenc (int): Time encoding method. 0: None, 1: Democritized time encoding, 2: Time2Vec encoding. Defaults to 1
      freq (str): Frequency of the data. Defaults to 'h'

    Returns:
      `np.array`: Time features
    """
    dates['month'] = dates.ts.dt.month
    dates['day'] = dates.ts.dt.day
    dates['weekday'] = dates.ts.dt.weekday
    dates['hour'] = dates.ts.dt.hour
    dates['minute'] = dates.ts.dt.minute
    dates['minute'] = dates.minute.map(lambda x: x//15)

    if timeenc==0:
        dates = dates.drop('ts', axis=1)
        return dates.values
    if timeenc==1:
        dates = dates.drop('ts', axis=1)
        data = []
        for col in dates.columns:
            data.append(dates[col].values)
        return np.stack(data, axis=-1)
