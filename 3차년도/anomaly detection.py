import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from models.ns_Transformer import Model
import torch
from models.DLinear import DLinear_TimeSeriesForecasting

# anaomaly detection
class AnomalyDetection:
    """
    Detect anomaly data points in dataset 

    Attributes:
    - data: Dataframe 
    - target_col: str
    - outliers_fraction: float
    - figsize: tuple

    Methods:
    - CART(): return model
    - visualize(result, anomaly_type): 
    """
    def __init__(self, data, target_col, outliers_fraction = float(0.03), figsize = (12,8)):
        self.data = data
        self.target_col = target_col
        self.figsize = figsize
        self.train_data = data.iloc[:int(len(data) * 0.6)]
        self.test_data = data.iloc[int(len(data) * 0.6):]
        scaler = StandardScaler()
        _scaled = scaler.fit_transform(self.train_data[target_col].values.reshape(-1,len(target_col)))
        self.scaled_train = pd.DataFrame(_scaled)
        _scaled_test = scaler.transform(self.test_data[target_col].values.reshape(-1,len(target_col)))
        self.scaled_test = pd.DataFrame(_scaled_test)
        self.outliers_fraction = outliers_fraction
    
    def CART(self, ):
        model = IsolationForest(contamination=self.outliers_fraction)
        model.fit(self.scaled_train.dropna())
        return model
        
    def STL(self):
        
        pass

    def ClusteringAD(self):
        pass

    def visualize(self, result, anomaly_type):
        fig, axs = plt.subplots(nrows = len(self.target_col), ncols = 1, figsize = self.figsize)
        anomaly_indices = []
        for i, column in enumerate(self.target_col):
            anomaly = result.loc[result['anomaly'] == -1, [column]]
            anomaly_indices.append(set(anomaly.index))
            axs[i].plot(result.index, result[column], color = 'black', linestyle = '--', label = 'Normal')
            axs[i].plot(anomaly.index, anomaly[column], 'ro', label = 'Anomaly')
            axs[i].set_ylabel(str(column))
            axs[i].legend()
            
            
            
        plt.suptitle(f'Outlier detection with {anomaly_type}')
        plt.savefig('anomaly_detection.png')
        plt.show()
        # Find common indices among all target columns
        common_anomaly_indices = set.intersection(*anomaly_indices)
        common_anomalies = result.loc[list(common_anomaly_indices)]

        return common_anomalies

    def get_output(self, anomaly_type):
        if anomaly_type == "CART":
            model = self.CART()
        elif anomaly_type == "STL":
            model = self.STL()
        elif anomaly_type == "Cluster":
            model = self.ClusteringAD()
        else:
            print("Invalid anomaly detection type")
        result = self.train_data.copy()
        print("prediction result", model.predict(self.scaled_train.dropna()))
        result['anomaly'] = model.predict(self.scaled_train.dropna())
        anomaly_data = self.visualize(result, anomaly_type)
        return anomaly_data
    
if __name__ == "__main__":
    from utils.dataloader import TimeSeriesDataset
    from models.configs import Config
    from torch.utils.data import DataLoader
    from models.ns_Transformer import ns_TimeSeriesForecasting




    # Load the dataset
    data_path = 'data/sbk_ad_B_v2.csv'
    data = pd.read_csv(data_path, index_col = 0)
    anomaly = AnomalyDetection(data, target_col = ['BP', 'H2S'])
    out = anomaly.get_output(anomaly_type = 'CART')
    out.drop(['anomaly'], axis = 1, inplace = True)
    out.to_csv('anomaly_sbk_B.csv')
    size = {
            'seq_len': 30,
            'label_len': 15,
            'pred_len': 1
        }

    enc_in = 16
    dec_in = 16
    c_out = 16
    freq = 'd'
    d_model = 16
    d_ff = 64
    config = Config(
        data_name = 'sbk',
        path= data_path,
        seq_len= size['seq_len'],
        label_len= size['label_len'],
        pred_len= size['pred_len'],
        variate= 'm',
        scale= True,
        is_timeencoded= True,
        random_state= 42,
        output_attention = False,
        enc_in = enc_in,
        d_model = d_model,
        embed = 'fixed',
        freq = freq,
        dropout = 0.05,
        dec_in = dec_in,
        factor = 1,
        n_heads = 8,
        d_ff = d_ff,
        activation = 'gelu',
        e_layers = 2,
        d_layers = 1,
        c_out = c_out,
        batch_size= 32,
        epoch= 20,
        lr= 0.00005,
        loss= 'mse',
        scheduler= 'exponential',
        inverse_scaling = False,
        num_workers = 0,
    )


    test_data = TimeSeriesDataset(
            path = config.path,
            split="train",
            seq_len=config.seq_len,
            label_len=config.label_len,
            pred_len=config.pred_len,
            scale=config.scale,
            is_timeencoded=config.is_timeencoded,
            frequency=config.freq,
            random_state=config.random_state,
        )

    test_dataloader = DataLoader(
                        test_data,
                        batch_size=config.batch_size,
                        shuffle=True,
                        num_workers=config.num_workers,
                        )
    
    # model = ns_TimeSeriesForecasting(config)
    train_data = data.iloc[:int(len(data) * 0.6)]
    mean = np.mean(train_data['BP'])
    std = np.std(train_data['BP'])
    
    PATH = '/Users/sujinchoi/Desktop/nsTransformer/weights/final_NSFormer-AD-20_30_15_1.ckpt'
    model = ns_TimeSeriesForecasting.load_from_checkpoint(PATH)
    # PATH = '/Users/sujinchoi/Desktop/nsTransformer/weights/final_DLinear-AD-20_30_15_1.ckpt'
    # model = DLinear_TimeSeriesForecasting.load_from_checkpoint(PATH)
    
    predictions, targets = model.predict(test_dataloader)

    # Convert anomaly indices to integer index positions
    anomaly_indices = out[out['BP'] <15000].index
    test_indices = anomaly.train_data.index
    integer_anomaly_indices = [test_indices.get_loc(idx) for idx in anomaly_indices if idx in test_indices]
    

    plt.figure(figsize=(30, 7))
    plt.scatter(integer_anomaly_indices, predictions.squeeze()[integer_anomaly_indices], color='red', label='Anomalies')
    plt.plot(targets.squeeze(), color = 'black', label='True', alpha = 0.5)
    plt.plot(predictions.squeeze(),color = 'blue',linestyle = '--', label='Pred')
    plt.legend()
    plt.show()

    # Modify 'total_in_vol' for outliers and predict 'BP'
    # 1. find the indices of x for each outlier

    # Convert indices to datetime
    anomaly_indices = pd.to_datetime(anomaly_indices)

    # Create a DataFrame to hold the previous 15 days for each index
    previous_15_days = pd.DataFrame({
        'anomaly_date': anomaly_indices
    })

    # Function to get previous 15 days
    def get_previous_15_days(date):
        return pd.date_range(end=date, periods=size['seq_len'] + 1)

    # Apply the function and create a new DataFrame
    previous_15_days_expanded = previous_15_days['anomaly_date'].apply(get_previous_15_days).explode().reset_index(drop=True)
    previous_15_days_expanded = pd.DataFrame({
        'previous_date': previous_15_days_expanded
    })
    previous_15_days_expanded.set_index('previous_date', inplace=True)

    # Ensure the index is in datetime format
    previous_15_days_expanded_indices = pd.to_datetime(previous_15_days_expanded.index)
    anomaly.train_data.index = pd.to_datetime(anomaly.train_data.index)
    outlier_x = anomaly.train_data.loc[previous_15_days_expanded.index]
    
    total_in_vol_range = np.linspace(train_data['total_in_vol'].min(), train_data['total_in_vol'].max(), 100)
    
    max_bp = -np.inf
    best_total_in_vol = None
    modified_df= pd.DataFrame(
        data = {
            'original_BP': anomaly.train_data['BP'][size['seq_len']:],
            'original_totalvol': anomaly.train_data['total_in_vol'][size['seq_len']:],
            'original_pred': predictions.squeeze() * std + mean
        },
        index = anomaly.train_data.index[size['seq_len']:],

    )
    original_data = anomaly.train_data.copy()
    for new_total_in_vol in total_in_vol_range:        
        # Modify the 'total_in_vol' for outliers in the dataset
        original_data.loc[previous_15_days_expanded_indices,'total_in_vol'] = new_total_in_vol
        tmp_path = 'modified_train_data.csv'
        original_data.to_csv(tmp_path)
        modified_data = TimeSeriesDataset(
            path = tmp_path,
            split="all",
            seq_len=config.seq_len,
            label_len=config.label_len,
            pred_len=config.pred_len,
            scale=config.scale,
            is_timeencoded=config.is_timeencoded,
            frequency=config.freq,
            random_state=config.random_state,
        )

        modified_data_loader = DataLoader(modified_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

        modified_predictions, _ = model.predict(modified_data_loader)        
        modified_df[f'BP_{new_total_in_vol}'] = modified_predictions * std + mean
    modified_df.loc[anomaly_indices]
    #     current_max_bp = modified_predictions.squeeze().max()
        
    #     if current_max_bp > max_bp:
    #         max_bp = current_max_bp
    #         best_total_in_vol = new_total_in_vol
    
    # print(f"The 'total_in_vol' that shows the maximum 'BP' is: {best_total_in_vol} with BP value: {max_bp}")


