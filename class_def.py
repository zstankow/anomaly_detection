import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", message=".*tf.losses.sparse_softmax_cross_entropy is deprecated.*")
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from tensorflow import keras
from keras.models import Model
from keras import layers
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
from keras.saving import register_keras_serializable
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
from extract_data import *
from logger import logger
import json
import pickle

SINGLE_DAY = 86400

@register_keras_serializable()
def custom_loss_mse(y_true, y_pred):
    gas_true = y_true[:, :, 0]
    gas_pred = y_pred[:, :, 0]
    return tf.reduce_mean(tf.square(gas_true - gas_pred))

def save_preprocessor_instance(file_path, instance):
    with open(f'{file_path}/processor.pkl', 'wb') as file:
        pickle.dump(instance, file)

def load_preprocessor_instance(file_path):
    with open(f'{file_path}/processor.pkl', 'rb') as file:
        loaded_preprocessor = pickle.load(file)
    return loaded_preprocessor

def read_file_contents(file_path):
    with open(file_path, 'r') as file:
        loaded_list = json.load(file)
    return loaded_list

def change_date_format(date_str, new_format):
    datetime_obj = datetime.strptime(date_str, '%Y-%m-%d')
    formatted_date = datetime_obj.strftime(new_format)
    return formatted_date

def compare_dates(date1, date2):
    datetime1 = datetime.strptime(date1, '%d-%m-%Y').date()
    datetime2 = datetime.strptime(date2, '%Y-%m-%d %H:%M:%S').date()
    return datetime1 < datetime2

def get_start_end_epochs(current_epoch, days_back, window_size):
    start_of_day_epoch, end_of_day_epoch = get_current_day(current_epoch) 
    start_epoch = start_of_day_epoch - days_back*SINGLE_DAY
    end_epoch = start_epoch + window_size*SINGLE_DAY
    return start_epoch, end_epoch, start_of_day_epoch, end_of_day_epoch

def get_max_anomaly_row(file_path):
    df = pd.read_csv(file_path)
    anomaly_df = df[df['Anomaly'] == 1]
    if anomaly_df.empty:
        return None
    max_mse_row = anomaly_df.loc[anomaly_df['MSE'].idxmax()]
    return max_mse_row 
    

def write_results_to_file(file_path, val1, val2, val3):
    try:
        with open(file_path, 'w') as f:
            f.write(f"{val1:.3f} {val2:.3f} {val3}")
    except Exception as err:
        print(err)


def get_current_day(current_epoch):
    current_datetime = datetime.fromtimestamp(current_epoch, tz=timezone.utc)
    start_of_day = datetime(current_datetime.year, current_datetime.month, current_datetime.day)
    end_of_day = start_of_day + pd.Timedelta(days=1)
    start_of_day_epoch = int(start_of_day.replace(tzinfo=timezone.utc).timestamp())
    end_of_day_epoch = int(end_of_day.replace(tzinfo=timezone.utc).timestamp())
    return start_of_day_epoch, end_of_day_epoch


def epoch_to_datetime_string(epoch_time):
    date_time = datetime.fromtimestamp(epoch_time, tz=timezone.utc)
    formatted_date = date_time.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_date


def epoch_to_date_string(epoch_time):
    dt = datetime.fromtimestamp(epoch_time, tz=timezone.utc)
    formatted_date = dt.strftime('%d-%m-%Y')
    return formatted_date

def get_date_string(file_path):
    try:
        with open(file_path, 'r') as file:
            date = file.read()
        return date
    
    except Exception as e:
        print(f"An error occurred: {e}")

def update_date_to_txt(file_path, date_string):
    try:
        with open(file_path, 'w') as file:
            file.write(date_string)

    except Exception as e:
        print(f"An error occurred: {e}")
    

def append_mse_to_csv(file_path, train_start_epoch, current_end_epoch, mse_value, threshold):
    # convert epoch to date
    end_date = datetime.fromtimestamp(current_end_epoch, tz=timezone.utc)
    two_weeks_prior_date = end_date - pd.Timedelta(weeks=2)
    two_weeks_prior_epoch = int(two_weeks_prior_date.timestamp())


    is_anomaly = mse_value > threshold
    new_record = pd.DataFrame([{'Train_Start_Epoch': train_start_epoch,
                                'Collection_End_Epoch': current_end_epoch,
                                'MSE': mse_value, 
                                'Anomaly': is_anomaly}])
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Train_Start_Epoch', 'Collection_End_Epoch', 'MSE', 'Anomaly'])
    
    df = pd.concat([df, new_record], ignore_index=True)
    df['Collection_End_Epoch'] = df['Collection_End_Epoch'].astype(int)
    df = df[df['Collection_End_Epoch'] >= two_weeks_prior_epoch]
    df.to_csv(file_path, index=False)
    return is_anomaly


def custom_loss_mse(y_true, y_pred):
    gas_true = y_true[:, :, 0]
    gas_pred = y_pred[:, :, 0]
    return tf.reduce_mean(tf.square(gas_true - gas_pred))


def build_model():
    input_shape = (80, 5)
    hidden1_dim = 64
    hidden2_dim = 32

    encoder_input = keras.Input(shape=input_shape)
    x = layers.Conv1D(hidden1_dim, kernel_size=3, activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    x = layers.Conv1D(hidden2_dim, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)

    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(hidden2_dim, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(size=2)(x)
    decoder_output = layers.Conv1D(5, kernel_size=3, activation='linear', padding='same')(x)

    autoencoder = Model(encoder_input, decoder_output)

    autoencoder.compile(optimizer='adam', loss=custom_loss_mse)
    return autoencoder


def lrfn(epoch):
    if epoch < 10:
        lr = 1e-3
    elif 10 <= epoch < 40:
        lr = 1e-4
    else:
        lr = 1e-5
    return lr


class DataFrameCleaner:
    def __init__(self):
        self.cleaning_train = True
        self.c = 2
        self.average_humidity_train = None

    def clean_df(self, df):
        df = df.reset_index(drop=True)
        df = self.standardize_df(df)
        df = df.sort_values(by=['Sensor', 'Step', 'Epoch'])
        segment_indices_code_12 = df[df['Code'] == -12].index.to_numpy()
        segment_indices_to_drop = np.concatenate(([0], segment_indices_code_12))
        df = df.drop(segment_indices_to_drop).reset_index(drop=True)
        df['Time_Diff'] = df['Epoch'].diff() > 1800
        df = self.remove_outliers(df)
        df = self.check_humidity_values(df)
        df = self.get_step_id(df)
        df = self.sort_df(df)
        df['Session'] = np.nan
        df['Slot'] = np.nan
        df = self.label_slot_session(df)
        df['Datetime'] = df['Epoch'].apply(self.epoch_to_date)
        df['Datetime']= pd.to_datetime(df['Datetime'])
        df.reset_index(inplace=True, drop=True)
        self.cleaning_train = False
        return df

    def check_humidity_values(self, df):
        if self.cleaning_train == True: # cleaning train data
            humid_is_0 = df[df['Humid'] == 0]
            humid_not_0 = df[df['Humid'] != 0]
            self.average_humidity_train = humid_not_0['Humid'].mean()
            self.std_humidity_train = humid_not_0['Humid'].std()
            if humid_is_0.empty:
                return df
            else:
                df.loc[df['Humid'] == 0, 'Humid'] = self.average_humidity_train
        else: # cleaning test data
            if self.average_humidity_train is not None: # aleady cleaned train data 
                humid_not_0 = df[df['Humid'] != 0]
                self.average_humidity_test = humid_not_0['Humid'].mean()
                df.loc[df['Humid'] == 0, 'Humid'] = self.average_humidity_train
            else: # did not clean train data
                humid_is_0 = df[df['Humid'] == 0]
                if humid_is_0.empty:
                    return df
                else:
                    humid_not_0 = df[df['Humid'] != 0]
                    average_humidity = humid_not_0['Humid'].mean()
                    df.loc[df['Humid'] == 0, 'Humid'] = average_humidity
        return df


    def standardize_df(self, df):
        col_names = ['Sensor', 'ID', 'Time', 'Epoch', 'Temp', 'Pressure', 'Humid', 'Gas', 'Step', 'Model', 'Lab', 'Code']
        df = df.iloc[:, :12]
        df.columns = col_names
        return df

    def remove_outliers(self, df):
        cleaned_segments = []
        segment_indices_time_diff = df[df['Time_Diff'] == True].index.to_numpy()
        segment_indices = np.concatenate(([0], segment_indices_time_diff, [len(df)]))

        for i in range(len(segment_indices) - 1):
            segment = df.loc[segment_indices[i]:segment_indices[i + 1] - 1]
            if not segment['Gas'].isnull().all():
                segment_cleaned = segment.copy()
                outliers = self.label_outliers(segment)
                segment_cleaned.loc[outliers, 'Gas'] = np.nan
                segment_cleaned['Gas'] = segment_cleaned['Gas'].bfill()
                segment_cleaned['Gas'] = segment_cleaned['Gas'].ffill()
                cleaned_segments.append(segment_cleaned)

        df_cleaned = pd.concat(cleaned_segments, ignore_index=False)
        return df_cleaned
    
    def label_outliers(self, segment):
        if not self.cleaning_train:
            segment_gas = np.log1p(segment['Gas'])
        else:
            segment_gas = segment['Gas']
        Q1 = segment_gas.quantile(0.25)
        Q3 = segment_gas.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = 0
        upper_bound = Q3 + self.c * IQR
        outliers = (segment_gas <= lower_bound) | (segment_gas > upper_bound)
        return outliers

    def get_step_id(self, df):
        df['Sensor_Step'] = df['Sensor'] * 10 + df['Step']
        return df

    def sort_df(self, df):
        sorted_df_model = df.sort_values(by=['Epoch', 'Time'])
        return sorted_df_model

    def label_slot_session(self, df):
        session = 0
        slots = 0
        slotEpoch = prevEpoch = df.iloc[0]['Epoch']
        for idx, row in df.iterrows():
            if row['Epoch'] - prevEpoch > 20 and row['Step'] == 0:
                session += 1
                prevEpoch = row['Epoch']
                if row['Epoch'] - slotEpoch > 30 * 60:
                    slots += 1
                    session = 0
                    slotEpoch = row['Epoch']
            df.at[idx, 'Slot'] = slots
            df.at[idx, 'Session'] = session
        return df

    def epoch_to_date(self, epoch_time):
        date_time = datetime.fromtimestamp(epoch_time, tz=timezone.utc)
        formatted_date = date_time.strftime('%Y-%m-%d %H:%M:%S')
        return formatted_date

    def list_to_df(self, data):
        data_df = pd.DataFrame(data, columns=['Gas', 'Temp', 'Humid', 'Pressure', 'Slot', 'Session', 'Sensor_Step'])
        return data_df


class DataPreprocessor:
    def __init__(self, df):
        self.df = df
        self.scalers = [StandardScaler() for _ in range(80)]
        self.df['Datetime'] = pd.to_datetime(self.df['Datetime'])
        self.trained_model = None
        self.train_new_session = True
        self.mse_train_data = None
        self.mse_new_data = None
        self.mse_new_data_fixed = None
        self.mse_new_data_fixed_start = None
        self.reconstructed_test_data = None
        self.reconstructed_test_data_fixed = None
        self.reconstructed_test_data_fixed_start = None
        self.reconstructed_train_data = None
        self.threshold = None
        self.session_0_means = None
        self.input_empty = False

    def save_fixed_start_model(self, file_path):
        try:
            self.trained_model.save(f'{file_path}/fixed_start_model.h5')
        except Exception as err:
            logger.error(f'{err}: Error occured in save_fixed_start_model')

    def save_fixed_model(self, file_path):
        try:
            self.trained_model.save(f'{file_path}/fixed_model.h5')
        except Exception as err:
            logger.error(f'{err}: Error occured in save_fixed_model')

    def reconstruct_new_data_fixed_start_model(self, file_path):
        try:
            fixed_model = tf.keras.models.load_model(f'{file_path}/fixed_start_model.h5', custom_objects={'custom_loss_mse': custom_loss_mse})
        except Exception as err:
            logger.error(f'{err}: Error occured in reconstruct_new_data_fixed_start_model')
        self.reconstructed_test_data_fixed_start = fixed_model.predict(self.test_data, verbose=False)
        self.mse_new_data_fixed_start = self.get_mean_mse(self.test_data, self.reconstructed_test_data_fixed_start)
    
    def reconstruct_new_data_fixed_model(self, file_path):
        try:
            fixed_model = tf.keras.models.load_model(f'{file_path}/fixed_model.h5', custom_objects={'custom_loss_mse': custom_loss_mse})
        except Exception as err:
            logger.error(f'{err}: reconstruct_new_data_fixed_model')
        self.reconstructed_test_data_fixed = fixed_model.predict(self.test_data, verbose=False)
        self.mse_new_data_fixed = self.get_mean_mse(self.test_data, self.reconstructed_test_data_fixed)  

    def preprocess_and_train(self):
        if self.train_new_session:
                self.preprocess(self.df)
                self.train_model_if_new_session()
                self.reconstruct_train_data()

    def preprocess(self, df):
        df = df.reset_index(drop=True)
        if self.train_new_session:
            self.session_0_means = [df.loc[(df['Session'] == 0) & (df['Sensor_Step'] == i), ['Gas', 'Temp', 'Humid']].mean().tolist() for i in range(80)]
            input = self.get_input_set_df(df)
            self.train_data = self.prepare_input(input)

        else:
            input = self.get_input_set_df(df)
            if not input.empty:
                self.test_data = self.prepare_input(input)
                self.reconstruct_new_data()
            else:
                self.input_empty = True
            self.test_data = self.prepare_input(input)
            self.reconstruct_new_data()
            self.mse_per_sensor_to_csv = self.get_array_mse_per_sensor()

    def get_input_set_df(self, df):
        input_data_list = []
        sub_list = []
        counter_dict = {}
        current_session = 0
        for index, row in df.iterrows():
            if (index > 0 and row['Session'] != current_session) or (index == len(df) - 1):
                current_session = row['Session']
                if len(counter_dict) != 80:
                    sub_list = self.fill_missing_sensor_steps(df.iloc[index - 1], counter_dict, sub_list, input_data_list)
                input_data_list.extend(sorted(sub_list, key=lambda x: x[-1]))
                counter_dict, sub_list = {}, []
            vsensor = row['Sensor_Step']
            if (vsensor not in counter_dict) and (index != len(df) - 1):
                counter_dict[vsensor] = 1
                sub_list.append([row['Gas'], row['Temp'], row['Humid'], row['Datetime'].date(), row['Slot'], row['Session'], row['Sensor_Step']])
        input_data_df_all = self.list_to_df(input_data_list)
        return input_data_df_all

    def prepare_input(self, df):
        data = df.drop(['Date', 'Session', 'Slot', 'Sensor_Step'], axis=1)
        data_scaled = self.scale_data_separate_sensors_no_folds_with_smoothing(data, sigma=2)
        return data_scaled

    def scale_data_separate_sensors_no_folds_with_smoothing(self, data, sigma=2):
        data_np = np.array(data, dtype=np.float64).reshape(-1, 80, 5)
        if self.train_new_session:
            for sensor in range(80):
                sensor_data = data_np[:, sensor, :]
                smoothed_sensor_data = gaussian_filter1d(sensor_data, sigma=sigma, axis=0)
                scaled_sensor_data = self.scalers[sensor].fit_transform(smoothed_sensor_data)
                data_np[:, sensor, :] = scaled_sensor_data

        else:
            for sensor in range(80):
              sensor_data = data_np[:, sensor, :]
              smoothed_sensor_data = gaussian_filter1d(sensor_data, sigma=sigma, axis=0)
              scaled_sensor_data = self.scalers[sensor].transform(smoothed_sensor_data)
              data_np[:, sensor, :] = scaled_sensor_data
        return data_np

    def list_to_df(self, data):
        data_df = pd.DataFrame(data, columns=['Gas', 'Temp', 'Humid', 'Date', 'Slot', 'Session', 'Sensor_Step'])
        data_df['Gas*Temp'] = data_df['Gas'] * data_df['Temp']
        data_df['Gas*Humid'] = data_df['Gas'] * data_df['Humid']
        return data_df

    def fill_missing_sensor_steps(self, row, counter, sub_data, data):
        total_missing = 80 - len(counter)
        sess = row['Session']
        slot = row['Slot']
        if total_missing < 20:
            for i in range(80):
                if i not in counter:
                    if not data:
                        vals_prev_sess = self.session_0_means[i]
                    elif sess == 0 and slot == 0:
                        vals_prev_sess = []
                    else:
                        vals_prev_sess = data[-80 + i][:3]
                    sub_data.append([*vals_prev_sess, row['Datetime'].date(), row['Slot'], row['Session'], i])
            return sub_data
        return []

    def train_model_if_new_session(self):
        model = build_model()
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = False)
        history = model.fit(self.train_data, self.train_data, epochs=50, batch_size=32, callbacks=[lr_callback], verbose=0)
        self.trained_model = model
        self.train_new_session = False

    def reconstruct_train_data(self):
        self.reconstructed_train_data = self.trained_model.predict(self.train_data, verbose=False)
        self.mse_train_data = self.get_mean_mse(self.train_data, self.reconstructed_train_data)
        self.std_train_data = np.std(np.square(self.train_data[:,:,0] - self.reconstructed_train_data[:,:,0]))
        self.threshold = self.mse_train_data + 4*self.std_train_data

        self.mse_train_per_sensor = self.get_mse_per_sensor(self.train_data, self.reconstructed_train_data)
        self.std_train_per_sensor =  np.std(np.square(self.train_data[:,:,0] - self.reconstructed_train_data[:,:,0]), axis=0)
        self.threshold_per_sensor = self.mse_train_per_sensor + 4*self.std_train_per_sensor

    def reconstruct_new_data(self):
        self.reconstructed_test_data = self.trained_model.predict(self.test_data, verbose=False)
        self.mse_new_data = self.get_mean_mse(self.test_data, self.reconstructed_test_data)
        self.mse_new_per_sensor = self.get_mse_per_sensor(self.test_data, self.reconstructed_test_data)
    
    def get_array_mse_per_sensor(self):
        anomalies = np.where(self.mse_new_per_sensor > self.threshold_per_sensor, self.mse_new_per_sensor, 0)
        return anomalies

    def get_mean_mse(self, data, reconstructed_data):
        mse_val = np.mean(np.square(data[:,:,0] - reconstructed_data[:,:,0]))
        return mse_val
    
    def get_mse_per_sensor(self, data, reconstructed_data):
        squared_diff = np.square(data[:,:,0] - reconstructed_data[:,:,0])
        mse_per_sensor = np.mean(squared_diff, axis=0)
        return mse_per_sensor

    

class Sensor:
    def __init__(self, bid, creation_date, days_back=4, window_size=3, start_fixed_window=3):
        self.bid = bid
        self.creation_date = creation_date
        self.date_fixed = None
        self.date = None
        self.prev_date = None
        date_only = creation_date.split(' ')[0]
        formatted_date_str = date_only.replace('-', '')
        self.uid = bid + '_' + formatted_date_str + '_' + str(days_back) + '_' + str(window_size) + '_' + str(start_fixed_window)
        self.file_path_date_fixed = f'{self.uid}/date_fixed.txt'
        self.file_path_date = f'{self.uid}/date.txt'
        self.days_back = days_back
        self.window_size = window_size
        self.start_fixed_window = start_fixed_window
    
    def initiate_connection(self, current_start_epoch, current_end_epoch):
        try:
            logger.info(f'{self.bid}: Initializing connection')
            start_fixed_model_date = pd.to_datetime(self.creation_date).tz_localize('UTC') + pd.Timedelta(days=self.start_fixed_window)
            current_date = datetime.fromtimestamp(current_start_epoch, tz=timezone.utc)
            self.date_fixed = epoch_to_date_string(current_start_epoch)
            train_start_epoch, _, start_of_day_epoch, _ = get_start_end_epochs(current_start_epoch, self.days_back, self.window_size)
            self.date = epoch_to_date_string(start_of_day_epoch)
            if start_fixed_model_date > current_date:
                logger.info(f'{self.bid}: Chosen start fixed model date {start_fixed_model_date} is after current detection date. Next detection will re-initialize script. System exit.')
                return self.date, -1, -1, -1, self.date_fixed, False, np.zeros(80), -1, -1, -1

            current_start_date = epoch_to_datetime_string(current_start_epoch)
            current_end_date = epoch_to_datetime_string(current_end_epoch)
            train_start_date = epoch_to_datetime_string(train_start_epoch)
            logger.info(f'{self.bid}: Received new data between {current_start_date} and {current_end_date}')

            csv_filepath = extract_and_save_data(board_id=self.bid, 
                                                 creation_date=self.creation_date, 
                                                 filepath=self.uid, 
                                                 csv_start_date=train_start_date,
                                                 csv_end_date=current_end_date)
            logger.info(f"{self.bid}: Extracted csv to {csv_filepath}")
            df = pd.read_csv(f'{csv_filepath}') 
            if df.empty:
                logger.info(f"{self.bid}: {csv_filepath} was empty")
                return self.date, -1, -1, -1, self.date_fixed, False, np.zeros(80), -1, -1, -1

            existing_data = df[(df['Epoch'] >= start_of_day_epoch - self.days_back*SINGLE_DAY) & (df['Epoch'] < start_of_day_epoch - (self.days_back-self.window_size)*SINGLE_DAY)]
            new_data = df[(df['Epoch'] >= current_start_epoch) & (df['Epoch'] <= current_end_epoch)]
            if existing_data.empty:
                logger.error(f"{self.bid}: existing_data in Sensor.initiate_connection was empty")
                return self.date, -1, -1, -1, self.date_fixed, False, np.zeros(80), -1, -1, -1
            if new_data.empty:
                logger.error(f"{self.bid}: new_data in Sensor.initiate_connection was empty")
                return self.date, -1, -1, -1, self.date_fixed, False, np.zeros(80), -1, -1, -1

            cleaner = DataFrameCleaner()
            cleaned_existing = cleaner.clean_df(existing_data)
            cleaned_new = cleaner.clean_df(new_data)
            logger.info(f'{self.bid}: Successfully cleaned data')
            avg_humidity_train, std_humidity_train, avg_humidity_test = cleaner.average_humidity_train, cleaner.std_humidity_train, cleaner.average_humidity_test

            preprocessor = DataPreprocessor(cleaned_existing)
            preprocessor.preprocess_and_train()
            logger.info(f'{self.bid}: Successfully trained moving window model')
            save_preprocessor_instance(self.uid, preprocessor)
            logger.info(f'{self.bid}: Saved preprocessor instance to {self.uid}/preprocessor.pkl')
            preprocessor.save_fixed_model(self.uid)
            logger.info(f'{self.bid}: Saved fixed model to {self.uid}/fixed_model.h5')

            preprocessor.preprocess(cleaned_new)
            if preprocessor.input_empty:
                return self.date, -1, -1, -1, self.date_fixed, False, np.zeros(80), -1, -1, -1
            
            if self.start_fixed_window != self.window_size:
                logger.info(f'{self.bid}: Training fixed start model on {self.start_fixed_window - 1} days of data.')
                train_start_epoch_fixed, _, start_of_day_epoch_fixed, _ = get_start_end_epochs(current_start_epoch,
                                                                                                self.start_fixed_window, 
                                                                                                self.start_fixed_window-1)
                train_start_date_fixed = epoch_to_datetime_string(train_start_epoch_fixed)
                csv_filepath_fixed = extract_and_save_data(board_id=self.bid, 
                                                    creation_date=self.creation_date, 
                                                    filepath=self.uid, 
                                                    csv_start_date=train_start_date_fixed,
                                                    csv_end_date=current_end_date)
                logger.info(f'{self.bid}: Extracted data for start fixed model.')
                df = pd.read_csv(f'{csv_filepath_fixed}') 
                existing_data_fixed = df[(df['Epoch'] >= start_of_day_epoch_fixed - self.start_fixed_window*SINGLE_DAY) & (df['Epoch'] < start_of_day_epoch - SINGLE_DAY)]
                new_data_fixed = df[(df['Epoch'] >= current_start_epoch) & (df['Epoch'] <= current_end_epoch)]
                cleaner = DataFrameCleaner()
                cleaned_existing_fixed = cleaner.clean_df(existing_data_fixed)
                cleaned_new_fixed = cleaner.clean_df(new_data_fixed)
                logger.info(f'{self.bid}: Cleaned data for start fixed model.')
                preprocessor_fixed = DataPreprocessor(cleaned_existing_fixed)
                preprocessor_fixed.preprocess_and_train()
                logger.info(f'{self.bid}: Preprocessed data and trained start fixed model.')
                preprocessor_fixed.preprocess(cleaned_new_fixed)
                mse_new_data_start_fixed = preprocessor_fixed.mse_new_data
                preprocessor_fixed.save_fixed_start_model(self.uid)
                logger.info(f'{self.bid}: Saved start fixed model to {self.uid}/fixed_start_model.h5')

            else:
                mse_new_data_start_fixed = preprocessor.mse_new_data
                preprocessor.save_fixed_start_model(self.uid)
                logger.info(f'{self.bid}: Saved start fixed model to {self.uid}/fixed_start_model.h5')

            threshold = preprocessor.threshold
            mse_new_data = preprocessor.mse_new_data
            mse_per_sensor = preprocessor.mse_per_sensor_to_csv
            logger.info(f'{self.bid}: MSE: {mse_new_data:.4f}')
            file_path = f'{self.uid}/mse_moving_window_history.csv'
            is_anomaly = append_mse_to_csv(file_path, train_start_epoch, current_end_epoch, mse_new_data, threshold)
            if is_anomaly:
                logger.info(f'{self.bid}: Anomaly detected!')

            update_date_to_txt(self.file_path_date_fixed, self.date_fixed)
            update_date_to_txt(self.file_path_date, self.date)
            logger.info(f'{self.bid}: Appended MSE value of {mse_new_data:.4f} to {file_path}')
            return self.date, mse_new_data, 0, mse_new_data_start_fixed, self.date_fixed, is_anomaly, mse_per_sensor, avg_humidity_train, std_humidity_train, avg_humidity_test
        except Exception as err:
            logger.error(f"{self.bid}: {err}: Failed to complete Sensor.initiate_connection")
            return self.date, -1, -1, -1, self.date_fixed, False, np.zeros(80), -1, -1, -1

    
    def continue_connection(self, current_start_epoch, current_end_epoch):
        try:
            self.date_fixed = get_date_string(file_path = self.file_path_date_fixed)
            train_start_epoch, _, start_of_day_epoch, _ = get_start_end_epochs(current_start_epoch, self.days_back, self.window_size)
            self.date = epoch_to_date_string(start_of_day_epoch)
            prev_date = get_date_string(file_path=self.file_path_date)

            current_start_date = epoch_to_datetime_string(current_start_epoch)
            current_end_date = epoch_to_datetime_string(current_end_epoch)
            train_start_date = epoch_to_datetime_string(train_start_epoch)

            logger.info(f'{self.bid}: Received new data between {current_start_date} and {current_end_date}')

            csv_filepath = extract_and_save_data(board_id=self.bid, 
                                                 creation_date=self.creation_date, 
                                                 filepath=self.uid, 
                                                 csv_start_date=train_start_date,
                                                 csv_end_date=current_end_date)
            logger.info(f"{self.bid}: Extracted csv to {csv_filepath}")
            df = pd.read_csv(f'{csv_filepath}') 
            if df.empty:
                logger.info(f"{self.bid}: {csv_filepath} was empty")
                return self.date, -1, -1, -1, self.date_fixed, False, np.zeros(80), -1, -1, -1

            existing_data = df[(df['Epoch'] >= start_of_day_epoch - self.days_back*SINGLE_DAY) & (df['Epoch'] < start_of_day_epoch - (self.days_back-self.window_size)*SINGLE_DAY)]
            new_data = df[(df['Epoch'] >= current_start_epoch) & (df['Epoch'] <= current_end_epoch)]
            if existing_data.empty:
                logger.error(f"{self.bid}: existing_data in Sensor.continue_connection was empty")
                return self.date, -1, -1, -1, self.date_fixed, False, np.zeros(80), -1, -1, -1
            if new_data.empty:
                logger.error(f"{self.bid}: new_data in Sensor.continue_connection was empty")
                return self.date, -1, -1, -1, self.date_fixed, False, np.zeros(80), -1, -1, -1

            cleaner = DataFrameCleaner()
            cleaned_existing = cleaner.clean_df(existing_data)
            cleaned_new = cleaner.clean_df(new_data)
            logger.info(f'{self.bid}: Successfully cleaned data')
            avg_humidity_train, std_humidity_train, avg_humidity_test = cleaner.average_humidity_train, cleaner.std_humidity_train, cleaner.average_humidity_test


            if prev_date != self.date:
                preprocessor = DataPreprocessor(cleaned_existing)
                preprocessor.preprocess_and_train()
                logger.info(f'{self.bid}: Successfully trained moving window model')
                save_preprocessor_instance(self.uid, preprocessor)
                logger.info(f'{self.bid}: Saved preprocessor instance to {self.uid}/preprocessor.pkl')
                update_date_to_txt(self.file_path_date, self.date)
                logger.info(f'{self.bid}: Successfully updated date "{self.date}" to {self.file_path_date}')

            else:
                preprocessor = load_preprocessor_instance(self.uid)
                logger.info(f'{self.bid}: Loaded preprocessor instance from {self.uid}/preprocessor.pkl')

            preprocessor.preprocess(cleaned_new)
            preprocessor.reconstruct_new_data_fixed_start_model(f'{self.uid}')   
            logger.info(f'{self.bid}: Successfully loaded start fixed model from {self.uid}/fixed_model.h5')        
            mse_new_data_fixed_start = preprocessor.mse_new_data_fixed_start
            mse_new_data = preprocessor.mse_new_data
            logger.info(f'{self.bid}: MSE: {mse_new_data:.4f}')
            threshold = preprocessor.threshold
            mse_per_sensor = preprocessor.mse_per_sensor_to_csv
            file_path = f'{self.uid}/mse_moving_window_history.csv'
            is_anomaly = append_mse_to_csv(file_path, train_start_epoch, current_end_epoch, mse_new_data, threshold) 
            logger.info(f'{self.bid}: Appended MSE value of {mse_new_data:.4f} to {file_path}')

            if is_anomaly:
                logger.info(f'{self.bid}: Anomaly detected!')

            row_data = get_max_anomaly_row(file_path)
            if row_data is not None:
                _, _, start_of_day_epoch_fixed, _ = get_start_end_epochs(row_data['Collection_End_Epoch'], self.days_back, self.window_size)
                prev_date_fixed = self.date_fixed
                self.date_fixed = epoch_to_date_string(start_of_day_epoch_fixed)
                if prev_date_fixed != self.date_fixed: # if date_fixed changed (newest anomaly is largest)
                    preprocessor.save_fixed_model(self.uid)
                    logger.info(f'{self.bid}: Successfully saved fixed model to {self.uid}/fixed_model.h5')
                    mse_new_data_fixed = preprocessor.mse_new_data
                    update_date_to_txt(self.file_path_date_fixed, self.date_fixed)
                    logger.info(f'{self.bid}: Successfully updated fixed date "{self.date_fixed}" to {self.file_path_date_fixed}')
                else: # if date_fixed did not change (older anomaly is largest)
                    preprocessor.reconstruct_new_data_fixed_model(self.uid)
                    logger.info(f'{self.bid}: Successfully loaded fixed model from {self.uid}/fixed_model.h5')
                    mse_new_data_fixed = preprocessor.mse_new_data_fixed

            else:
                mse_new_data_fixed = 0
                if compare_dates(self.date_fixed, self.creation_date):
                        new_date_fixed = change_date_format(self.creation_date.split(" ")[0], "%d-%m-%Y")
                        update_date_to_txt(self.file_path_date_fixed, new_date_fixed)

            return self.date, mse_new_data, mse_new_data_fixed, mse_new_data_fixed_start, self.date_fixed, is_anomaly, mse_per_sensor, avg_humidity_train, std_humidity_train, avg_humidity_test
    
        except Exception as err:
            logger.error(f"{self.bid}: {err}: Failed to complete Sensor.continue_connection")
            self.date = epoch_to_date_string(start_of_day_epoch)
            return self.date, -1, -1, -1, self.date_fixed, False, np.zeros(80), -1, -1, -1
        

class EpochProcessor:
    def __init__(self, creation_date, end_date, temp_file_path, days_back=4, window_size=3, start_fix=3):
        self.creation_date = creation_date
        self.end_date = end_date
        self.temp_file_path = temp_file_path
        self.days_back = days_back
        self.start_fix = start_fix
        self.window_size = window_size
        # if self.start_fix > self.window_size:
        #     self.days_back = self.start_fix

    def get_current_day(self, current_date):
        start_of_day = datetime(current_date.year, current_date.month, current_date.day)
        end_of_day = start_of_day + pd.Timedelta(days=1)

        start_of_day_epoch = int(start_of_day.replace(tzinfo=timezone.utc).timestamp())
        end_of_day_epoch = int(end_of_day.replace(tzinfo=timezone.utc).timestamp())
        return start_of_day_epoch, end_of_day_epoch

    def get_epochs(self, df):
        self.creation_date = pd.to_datetime(self.creation_date)
        self.end_date = pd.to_datetime(self.end_date)
        current_date = self.creation_date + pd.Timedelta(days=self.days_back)
        rows = []

        while current_date <= self.end_date:
            start_of_current_day, end_of_current_day = self.get_current_day(current_date)
            new_data = df[(df['Epoch'] >= start_of_current_day) & (df['Epoch'] <= end_of_current_day)]
            if new_data.empty:
                current_date += pd.Timedelta(days=1)
                continue

            cleaner = DataFrameCleaner()
            cleaner.cleaning_train = False
            cleaned_new = cleaner.clean_df(new_data)

            slots = cleaned_new['Slot'].unique()
            new_data_by_slot = [cleaned_new[cleaned_new['Slot'] == slot_val] for slot_val in slots]

            for slot_data in new_data_by_slot:
                if not slot_data.empty:
                    first_row = slot_data.iloc[0]
                    last_row = slot_data.iloc[-1]
                    rows.append([first_row['Epoch'], last_row['Epoch']])
            
            print(f"Adding epochs to csv from date: {current_date.strftime('%Y-%m-%d')}")
            current_date += pd.Timedelta(days=1)
            
        result_df = pd.DataFrame(rows, columns=['StartEpoch', 'EndEpoch'])
        file_path_csv = f'{self.temp_file_path}/epochs.csv'
        result_df.to_csv(file_path_csv, index=False)
        return file_path_csv

    def get_epochs_csv(self, bid):
        csv_filepath = extract_and_save_data(board_id=bid, 
                                             creation_date=self.creation_date, 
                                             filepath=self.temp_file_path, 
                                             csv_start_date=self.creation_date,
                                             csv_end_date=self.end_date)
        print('Extracted csv from DB.')
        df = pd.read_csv(csv_filepath)
        return self.get_epochs(df)
        