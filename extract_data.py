import pyodbc
import csv
import os
from logger import logger
from datetime import datetime, timezone
import time
import pandas as pd

# Database connection details
DB_SERVER = ''
DB_USER = ''
DB_PWD = ''
DB_NAME = ''

def remove_duplicates(filename):
    df = pd.read_csv(filename)
    df = df.loc[(df.shift() != df).any(axis=1)]
    df.to_csv(filename, index=False)
    return filename

def format_date(date_string):
    return datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S").strftime("%d%m%YT%H%M%S")

def format_title_component(component):
    formatted_component = ""
    previous_char_was_space = True

    for char in component:
        if char.isspace():
            previous_char_was_space = True
        elif previous_char_was_space:
            formatted_component += char.upper()
            previous_char_was_space = False
        else:
            formatted_component += char
    return formatted_component

def format_date(date_string):
    return datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S").strftime("%d%m%YT%H%M%S")

def extract_and_save_data(board_id=None, creation_date=None, csv_start_date=None, csv_end_date=None, filepath=None):

    connection_string = f"DRIVER={{SQL Server}};SERVER={DB_SERVER};DATABASE={DB_NAME};UID={DB_USER};PWD={DB_PWD}"

    try:
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()

        cursor.execute(f"SELECT Location, Comment, House, InHouseLoc FROM Locations WHERE BOARDID = '{board_id}' AND dateCreated >= '{creation_date}'")
        location_data = cursor.fetchone()

        if not location_data:
            logger.error(f"No location data for BID {board_id} and creation date {creation_date}")
            raise ValueError("No location data for this BOARDID")


        location, comment, house, inhouse_loc = map(format_title_component, location_data)
        output_file_name = f"{board_id}_{format_date(creation_date)}_{location}_{comment}_{house}_{inhouse_loc}.csv"
        
        #full output path
        if filepath:
            os.makedirs(filepath, exist_ok=True)
            time.sleep(5)
            full_output_path = os.path.join(filepath, output_file_name)
        else:
            full_output_path = output_file_name


        selected_columns = [
            "[Sensor Index] AS Sensor",
            "[Sensor ID] AS ID",
            "[Time Since PowerOn] AS Tim",
            "[Real time clock] AS Epoch",
            "[Temperature] AS Temp",
            "[Pressure] AS Pressure",
            "[Relative Humidity] AS Humid",
            "[Resistance Gassensor] AS Gas",
            "[Heater Profile Step Index] AS Step",
            "[Scanning enabled] AS Mod",
            "[Label Tag] AS Lab",
            "[Error Code] AS Code"
        ]


        csv_date1 = datetime.strptime(csv_start_date, '%Y-%m-%d %H:%M:%S')
        cvs_start_epochtime = int(csv_date1.replace(tzinfo=timezone.utc).timestamp())
        csv_date2 = datetime.strptime(csv_end_date, '%Y-%m-%d %H:%M:%S')
        cvs_end_epochtime = int(csv_date2.replace(tzinfo=timezone.utc).timestamp())

        sql_query = f"SELECT {', '.join(selected_columns)} FROM DSData WHERE BOARDID = '{board_id}' AND dateCreated >= '{creation_date}' AND [Real time clock] >= '{cvs_start_epochtime}' AND [Real time clock] <= '{cvs_end_epochtime}' ORDER BY [Sensor Index], [Heater Profile Step Index], [Real time clock]"
        cursor.execute(sql_query)
        records = cursor.fetchall()

        # else:
        #     sql_query = f"SELECT {', '.join(selected_columns)} FROM DSData WHERE BOARDID = '{board_id}' AND dateCreated >= '{creation_date}' ORDER BY [Sensor Index], [Heater Profile Step Index], [Real time clock]"
        #     cursor.execute(sql_query)
        #     records = cursor.fetchall()

        with open(full_output_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([column.split(" AS ")[1] for column in selected_columns])
            csv_writer.writerows(records)

        full_output_path = remove_duplicates(full_output_path)
        return full_output_path

    except Exception as err:
        print(err)

    finally:
        if 'conn' in locals():
            conn.close()
    
