import sys
import os
from class_def import *
import warnings
warnings.filterwarnings("ignore")
import json
import argparse
import shutil

# python day_to_day.py E05A1B0CEE60 "2024-04-10" 1713157208 1713158999 filename.json

def retrieve_excel_data(vals):
    data = {
        "Detection_Date": pd.to_datetime(vals[0], format='%d-%m-%Y', errors='coerce'),
        "MSE": vals[1],
        "MSE_Fixed": vals[2],
        "MSE_Fixed_Start": vals[3],
        "Date_Fixed": pd.to_datetime(vals[4], format='%d-%m-%Y', errors='coerce'),
        "is_anomaly": vals[5],
        "Average_Humidity_Train": vals[7],
        "STD_Humidity_Train": vals[8],
        "Average_Humidity_Test": vals[9]
    }

    sensor_values = vals[6].flatten().tolist()
    for i in range(80):
        data[f'Sensor {i}'] = sensor_values[i]
    return data

def retrieve_return_file_data(vals):
    data = {        
        "Detection_Date": vals[0],
        "MSE": vals[1],
        "MSE_Fixed": vals[2],
        "MSE_Fixed_Start": vals[3],
        "Date_Fixed": vals[4],
        "is_anomaly": bool(vals[5]),
        "Average_Humidity_Train": vals[7],
        "STD_Humidity_Train": vals[8],
        "Average_Humidity_Test": vals[9],
        "MSE_per_sensor": (vals[6].flatten()).tolist()
    }
    return data

def delete_folder(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)

def get_df_columns():
    sensor_columns = [f'Sensor {i}' for i in range(80)]
    columns = ['Detection_Date', 'MSE', 'MSE_Fixed', 'MSE_Fixed_Start', 'Date_Fixed', 'is_anomaly', 'Average_Humidity_Train', 'STD_Humidity_Train', 'Average_Humidity_Test'] + sensor_columns
    return columns

def main():
    parser = argparse.ArgumentParser(description="Process sensor data and save results to a JSON file.\nExample usage:\npython script.py E05A1B0CEE60 '2024-04-10 00:00:00' 1713157208 1713158999 filename.txt")
    parser.add_argument('UID', type=str, help='Unique identifier for the sensor')
    parser.add_argument('CreationDate', type=str, help='Creation date of the sensor in "YYYY-MM-DD HH:MM:SS" format')
    parser.add_argument('StartEpoch', type=int, help='Start epoch time')
    parser.add_argument('EndEpoch', type=int, help='End epoch time')
    parser.add_argument('Filename', type=str, help='Output filename (including path if necessary)')

    parser.add_argument('--days_back', type=int, default=4, help='Number of days back from the detection for moving window model')
    parser.add_argument('--window', type=int, default=3, help='Number of training days')
    parser.add_argument('--start_fixed', type=int, default=3, help='Number of training days for the start fixed model')

    args = parser.parse_args()
    try:
        bid = args.UID
        creation_date = args.CreationDate
        creation_datetime = creation_date + ' 00:00:00'
        start_epoch = args.StartEpoch
        end_epoch = args.EndEpoch
        filename = args.Filename
        formatted_date_str = creation_date.replace('-', '')
        days_back = args.days_back
        window_size = args.window
        start_fixed = args.start_fixed
        if window_size > days_back:
            print("Error: Window size should be less than number of days back.")
            sys.exit(1)

    except Exception as err:
        print(err)
        sys.exit(1)
        
    folder_path = bid + '_' + formatted_date_str + '_' + str(days_back) + '_' + str(window_size) + '_' + str(start_fixed)
    if not os.path.exists(folder_path):
        sensor = Sensor(bid=bid, 
                        creation_date=creation_datetime, 
                        days_back=days_back, 
                        window_size=window_size, 
                        start_fixed_window=start_fixed)
        vals = sensor.initiate_connection(current_start_epoch=start_epoch,
                                            current_end_epoch=end_epoch)
        
        if vals[1] == -1:
            data = retrieve_return_file_data(vals)
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
            sys.exit(1)

        data = retrieve_excel_data(vals)
        columns = get_df_columns()
        df = pd.DataFrame([data], columns=columns)
        df.to_csv(f'{folder_path}/results.csv', mode='a', header=True, index=False)

    else:
        sensor = Sensor(bid=bid, 
                        creation_date=creation_datetime, 
                        days_back=days_back, 
                        window_size=window_size, 
                        start_fixed_window=start_fixed)
        vals = sensor.continue_connection(current_start_epoch=start_epoch,
                                            current_end_epoch=end_epoch)

        data = retrieve_excel_data(vals)
        columns = get_df_columns()
        df = pd.DataFrame([data], columns=columns)
        df.to_csv(f'{folder_path}/results.csv', mode='a', header=False, index=False)

    data = retrieve_return_file_data(vals)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    try:
        main()
    except SystemExit as err:
        pass
