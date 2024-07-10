# Anomoly Detection with Autoencoders

These two scripts process sensor data and calculate the MSE for three different models: moving window model, fixed model, and start fixed model. 

The script `day_to_day.py` is designed for autonomous daily detection. The results are saved to a JSON file and are updated within a CSV file. The other script `run_all.py` is designed for cycles that have already completed. The results are saved to a CSV file `results.csv`, which contains information such as the MSE values for the moving window model, the fixed model, and the fixed start model, as well as information about environmental conditions. In addition, the results file contains MSE values of each individual sensor as well as values pertaining to environmental conditions which allows for explainability of results and more insight into what caused the anomaly to occur. 

An example plot of the MSE for different models.

![image](https://github.com/zstankow/anomaly_detection/assets/150588332/514fbd72-8ab9-4d1e-a661-e50b0689739b)


## Usage of day_to_day.py script

To run the day-to-day script, execute it with the following command:

`python day_to_day.py <BID> <CreationDate> <StartEpoch> <EndEpoch> <Filename> [--days_back <DaysBack>] [--window <Window>] [--start_fixed <StartFixed>]`

__Arguments__
- `<BID>`: Unique identifier for the sensor.
- `<CreationDate>`: Creation date of the sensor in "YYYY-MM-DD" format.
- `<StartEpoch>`: Start epoch time.
- `<EndEpoch>`: End epoch time.
- `<Filename> `: Name of output file, can also be a path.

__Optional Arguments__

- `--days_back <DaysBack>` (default: 4): Number of days back from the detection for moving window model.
- `--window <Window>` (default: 3): Number of training days for the moving window model.
- `--start_fixed <StartFixed>` (default: 3): Number of days from the CreationDate to train the start fixed model<sup>[1](#note1)</sup>. Number of training days is StartFixed - 1.



### Example
```bashrc
python day_to_day.py E05A1B0CEE60 2024-04-10 1713157208 1713158999 filename.json --days_back 5 --window 4 --start_fixed 4
```

This command will process sensor data for the specified time range and save the results to `filename.json`. It is also acceptable to use  filename.txt instead of filename.json. Results of each detection will also be updated in `results.csv`.


## Usage of run_all.py script

To run the manual detection script, execute it with the following command:

`python run_all.py <BID> <CreationDate> <EndDate> [--days_back <DaysBack>] [--window <Window>] [--start_fixed <StartFixed>]`

__Arguments__
- `<BID>`: Unique identifier for the sensor.
- `<CreationDate>`: Creation date of the sensor in "YYYY-MM-DD" format.
- `<EndDate>`: End date of the sensor in "YYYY-MM-DD" format (user may choose an earlier date as well).

__Optional Arguments__

- `--days_back <DaysBack>` (default: 4): Number of days back from the detection for moving window model.
- `--window <Window>` (default: 3): Number of training days for the moving window model.
- `--start_fixed <StartFixed>` (default: 3): Number of training days for the start fixed model<sup>[1](#note1)</sup>.

### Example
```bashrc
python run_all.py E05A1B0CEE60 2024-04-10 2024-05-22 --days_back 5 --window 4 --start_fixed 4
```
The `CreationDate` must be the correct start date of the cycle, however the end date may vary according to the user's preference. 
This command will process sensor data for the specified time range and save the results to `results.csv`. 

## Output

The scripts will create a folder named `<BID>_<YYYYMMDD>_<DaysBack>_<Window>_<StartFixed>` (based on the sensor's BID and creation date). This folder will contain 8 files:
  - `BID_CreationDate_Location_Comment_House_InHouse_Loc.csv`: sensor data extracted from database, ranging from the beginning of the training set to the present detection
  - `date.txt`: date of most recent detection (training of moving window model is 4-3 from this date)
  - `date_fixed.txt`: date of last fixed model (training of fixed model is 4-3 from this date)
  - `preprocessor.pkl`: pickle file containing most recent instance of DataPreprocesser
  - `fixed_model.h5`: saved fixed model, trained <`days_back`-`window`> from date in date_fixed.txt (default is 4-3)
  - `fixed_start_model.h5`: saved fixed start model, trained on first <`start_fixed`> days of data (default is 3)
  - `mse_moving_window_history.csv`: contains MSE values of moving window model of last two weeks. From these values the fixed date is selected as the date with the largest __anomolous__ MSE value. 
  - `results.csv`: contains the MSE values of the moving window, fixed, and start fixed models, as well as the detection date and the fixed date for each detection. Contains the mean and standard deviation of humidity for the training set and the mean of humidity for the new detection. Also contains the MSE per sensor.

## Dependencies

- Python 3.11.9 and requirements.txt
```bashrc
pip install -r requirements.txt
```

## Overview of algorithm

![image](https://github.com/YieldX/ITC/assets/150588332/f2c0623a-64d3-4aaf-b185-64aed70e686a)


<a name="note1">[1]</a> Note: If the `start_fixed` value is greater than the value of `window`, the moving window model will not be initialized until the date of the detection is the same as the desired start fixed model date. 

For example, for BID E05A1B0CEE60 the creation date is April 10th. If the user chooses start_fixed to be 10, then the script will actually initialize only when the StartEpoch and EndEpoch received occur on April 20th (April 10 + 10 days). 
