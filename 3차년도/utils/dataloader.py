import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from timefeatures import time_features
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    def __init__(
            self,
            path,
            split = 'train',
            seq_len = 24 * 4 * 4,
            label_len = 24 * 4,
            pred_len = 24 * 4, 
            scale = True,
            random_state = 42,
            is_timeencoded = True,
            frequency = 'd'
            ):
        assert split in ['train', 'val', 'test']
        self.path = path 
        self.split = split
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.scale = scale
        self.random_state = random_state
        self.is_timeencoded = is_timeencoded
        self.frequency = frequency

        self.scaler = StandardScaler()

        self.prepare_data()

    def prepare_data(self):
        df = pd.read_csv(self.path) # 1st col: date 
        df['Date'] = pd.to_datetime(df['Date'])

        indices = df.index.tolist()
        train_size = 0.5
        val_size = 0.3
        test_size = 0.2

        train_end = int(len(indices) * train_size)
        val_end = train_end + int(len(indices) * val_size)
        train_indices = indices[:train_end]
        if self.split == 'train':
            split_indices = train_indices
        elif self.split == 'val':
            split_indices = indices[train_end:val_end]
        if self.split == 'test':
            split_indices = indices[val_end:]
            
        df_split = df.loc[split_indices]

        data_columns = df_split.columns[1:]
        data = df_split[data_columns]
        # data_y = df_split[data_columns[4:]] # y: exclude input columns
        self.feature_names = data_columns

        data = torch.FloatTensor(data.values)

        if self.scale:
            train_data = df.loc[train_indices][self.feature_names].values
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)
        else:
            data = data.values
        data_y =data[:,4:] # y: exclude input columns

        timestamp = df_split[['Date']]
        if not self.is_timeencoded:
            timestamp['month'] = timestamp.date.apply(lambda row: row.month, 1)
            timestamp['day'] = timestamp.date.apply(lambda row: row.day, 1)
            timestamp['weekday'] = timestamp.date.apply(lambda row: row.weekday(), 1)
            if self.frequency == 'h' or self.frequency == 't':
                timestamp['hour'] = timestamp.date.apply(lambda row: row.hour, 1)
            if self.frequency == 't':
                timestamp['minute'] = timestamp.date.apply(lambda row: row.minute, 1)
                timestamp['minute'] = timestamp.minute.map(lambda x: x // 15)
            timestamp_data = timestamp.drop('date', axis = 1).values
        else:
            timestamp_data = time_features(pd.to_datetime(timestamp.Date.values), freq = self.frequency)
            timestamp_data = timestamp_data.transpose(1,0)

        self.time_series_x = torch.FloatTensor(data)
        self.time_series_y = torch.FloatTensor(data_y)
        self.timestamp = torch.FloatTensor(timestamp_data)
    
    def __getitem__(self, index):
        x_begin_index = index
        x_end_index = x_begin_index + self.seq_len
        y_begin_index = x_end_index - self.label_len
        y_end_index = y_begin_index + self.label_len + self.pred_len

        x = self.time_series_x[x_begin_index:x_end_index]
        y = self.time_series_y[y_begin_index:y_end_index]
        x_timestamp = self.timestamp[x_begin_index:x_end_index]
        y_timestamp = self.timestamp[y_begin_index:y_end_index]

        return x, y, x_timestamp, y_timestamp

    def __len__(self):
        return len(self.time_series_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        if data.ndim >=3:
            data_shape = data.shape
            data = data.reshape(-1, data.shape[-1])
            data = self.scaler.inverse_transform(data.cpu().detach().numpy())
            data = torch.tensor(data).reshape(*data_shape)

        return data

    @property
    def num_features(self):
        return self.time_series_x.shape[1]

    @property
    def columns(self):
        return self.feature_names
    
if __name__ == "__main__":
    path = "sbk_ad_B.csv"
    train_data = TimeSeriesDataset(
            path = path,
            split="train",
            seq_len=30,
            label_len=30,
            pred_len=30,
        )
    train_dataloader = DataLoader(
                    train_data,
                    batch_size=32,
                    shuffle=True,
                    )


