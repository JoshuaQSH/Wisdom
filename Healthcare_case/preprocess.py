import os, sys
import numpy as np
import pandas as pd
from typing import List, Optional, Dict


# -----------
# 1. Healthcare Data Helper: function to compute HRV from heart rate data
# -----------
def compute_hrv_from_heart_rate(heart_rates: List[float], window_size: int = 15, jitter_std: float = 5.0, 
                           use_jitter: bool = False, timestamps: Optional[List[str]] = None) -> List[Dict]:

    def rr_intervals(hr_segment: List[float]) -> np.ndarray:
        return np.array([60000 / hr for hr in hr_segment if hr > 0])

    def rr_intervals_with_jitter(hr_segment: List[float], jitter_std: float = 5.0) -> np.ndarray:
        base_rr = rr_intervals(hr_segment)
        jitter = np.random.normal(0, jitter_std, size=len(base_rr))
        return base_rr + jitter

    def hrv_metrics(rr: np.ndarray) -> Dict[str, float]:
        rr_diffs = np.diff(rr)
        rmssd = np.sqrt(np.mean(rr_diffs ** 2)) if len(rr_diffs) > 0 else np.nan
        sdnn = np.std(rr, ddof=1) if len(rr) > 1 else np.nan
        return {
            "rmssd": round(rmssd, 2),
            "sdnn": round(sdnn, 2)
        }

    results = []
    for i in range(0, len(heart_rates) - window_size + 1):
        segment = heart_rates[i:i + window_size]
        if use_jitter:
            rr = rr_intervals_with_jitter(segment, jitter_std)
        else:
            rr = rr_intervals(segment)
        metrics = hrv_metrics(rr)
        metrics["second"] = i
        if timestamps:
            metrics["start_timestamp"] = timestamps[i]
        results.append(metrics)

    return results

# --------------
# 1. Function to preprocess healthcare data
# --------------
def preprocess_healthcare_data(data_path):
    health_care_path = os.path.join(data_path, 'Healthcare/data.csv')
    nurses_pd = pd.read_csv(health_care_path)
    print('The initial data contains {} rows and {} columns.'.format(nurses_pd.shape[0], nurses_pd.shape[1]))

    # Data Preprocessing (drop unnecessary columns and convert label to int)
    data_pd = nurses_pd.drop(columns=['X', 'Y', 'Z', 'EDA', 'TEMP', 'id'])
    print('The data we are interested in contains {} rows and {} columns.'.format(data_pd.shape[0], data_pd.shape[1]))

    # Calculate time difference between consecutive samples (in seconds), drop the first NaN and calculate average time
    # Ensure the datetime column is of datetime type
    data_pd['datetime'] = pd.to_datetime(data_pd['datetime'])
    data_pd = data_pd.sort_values(by='datetime')
    data_pd['time_diff'] = data_pd['datetime'].diff().dt.total_seconds()
    data_pd = data_pd.dropna(subset=['time_diff'])
    average_time_diff = data_pd['time_diff'].mean()
    data_pd.drop(columns=['time_diff'], inplace=True)

    # Calculate frequency (Hz)
    sampling_frequency_hz = 1 / average_time_diff
    print(f'Average Time Between Samples: {average_time_diff:.6f} seconds')
    print(f'Sampling Frequency: {sampling_frequency_hz:.2f} Hz')
    
    data_pd.set_index('datetime', inplace=True)
    resampled_df = data_pd.resample('1S').mean(numeric_only=True)
    resampled_df = resampled_df.dropna(subset=['HR'])
    resampled_df['label'] = resampled_df['label'].astype(int)
    resampled_df.to_csv(os.path.join(data_path, 'Healthcare/resampled_data.csv'), index=True)
    print('The data after resampling contains {} rows and {} columns.'.format(resampled_df.shape[0], resampled_df.shape[1]))
    
    data_pd = pd.read_csv(os.path.join(data_path, 'Healthcare/resampled_data.csv'))
    heart_rates = data_pd['HR'].value_counts()
    labels = data_pd['label'].value_counts()

    # Simple Statistical Analysis
    mean_heart_rate = data_pd['HR'].mean()
    median_heart_rate = data_pd['HR'].median()
    std_heart_rate = data_pd['HR'].std()

    print('The mean heart rate is {:.2f} bpm.'.format(mean_heart_rate))
    print('The median heart rate is {:.2f} bpm.'.format(median_heart_rate))
    print('The standard deviation of heart rate is {:.2f} bpm.'.format(std_heart_rate))
    
    # Create a new column for HRV, calculate the difference between consecutive heart rates every 15 seconds
    step = 15

    # New HRV column to be filled
    hrv_values = []

    for i in range(0, len(data_pd), step):
        chunk = data_pd.iloc[i:i+step]
        if len(chunk) == step:
            hr_list = chunk['HR'].tolist()
            ts_list = chunk['datetime'].astype(str).tolist()
            hrv_result = compute_hrv_from_heart_rate(hr_list, timestamps=ts_list, use_jitter=True)
            print(hrv_result)
            hrv_rmssd = hrv_result[0]['rmssd']
            hrv_values.extend([hrv_rmssd] * 15)
        else:
            hrv_values.extend([np.nan] * len(chunk))

    # Assign to new column
    data_pd['HRV'] = hrv_values

    # Export the data and delete downsampled data
    data_pd = data_pd[['HR', 'HRV', 'datetime', 'label']]
    data_pd.to_csv(os.path.join(data_path, 'Healthcare/hrv.csv'), index=False)
    print('The data with HRV contains {} rows and {} columns.'.format(data_pd.shape[0], data_pd.shape[1]))

    # Detete the resampled data
    os.remove(os.path.join(data_path, 'Healthcare/resampled_data.csv'))


# --------------
# 2. Function to preprocess Heartrate data
# --------------
def preprocess_heartrate_data(data_path):
    """
    Train_Data:
        - time_domain_features_train.csv - This file contains all time domain features of heart rate for training data
        - frequency_domain_features_train.csv - This file contains all frequency domain features of heart rate for training data
        - heart_rate_non_linear_features_train.csv - This file contains all non linear features of heart rate for training data
    Test_Data:
        - time_domain_features_test.csv - This file contains all time domain features of heart rate for testing data
        - frequency_domain_features_test.csv - This file contains all frequency domain features of heart rate for testing data
        - heart_rate_non_linear_features_test.csv - This file contains all non linear features of heart rate for testing data
    """
    heartrate_path = os.path.join(data_path, 'Heart_Rate_Prediction')
    train_path = os.path.join(heartrate_path, 'train_data')
    test_path = os.path.join(heartrate_path, 'test_data')
    
    if not os.path.exists(train_path):
        os.makedirs(train_path, exist_ok=True)  # exist_ok=True avoids a race condition
    if not os.path.exists(test_path):
        os.makedirs(test_path, exist_ok=True)

    # Columns to drop
    columns_to_drop_linear = ['SD1', 'SD2', 'sampen', 'higuci', 'datasetId']
    columns_to_drop_time = ['MEAN_RR','MEDIAN_RR','SDRR','SDSD','SDRR_RMSSD','pNN25','pNN50','KURT','SKEW','MEAN_REL_RR','MEDIAN_REL_RR','SDRR_REL_RR',
                  'RMSSD_REL_RR','SDSD_REL_RR','SDRR_RMSSD_REL_RR','KURT_REL_RR','SKEW_REL_RR']
    
    # Data path 
    train_non_linear_path  = os.path.join(data_path,  'Heart_Rate_Prediction/train_data/heart_rate_non_linear_features_train.csv')
    train_time_domain_path = os.path.join(data_path, 'Heart_Rate_Prediction/train_data/time_domain_features_train.csv')

    test_non_linear_path  = os.path.join(data_path,  'Heart_Rate_Prediction/test_data/heart_rate_non_linear_features_test.csv')
    test_time_domain_path = os.path.join(data_path, 'Heart_Rate_Prediction/test_data/time_domain_features_test.csv')
    
    # Pipeline
    train_non_linear_df  = pd.read_csv(train_non_linear_path)
    train_time_domain_df = pd.read_csv(train_time_domain_path)
    print('The initial non linear  train data contains {} rows and {} columns.'.format(train_non_linear_df.shape[0], train_non_linear_df.shape[1]))
    print('The initial time domain train data contains {} rows and {} columns.'.format(train_time_domain_df.shape[0], train_time_domain_df.shape[1]))
    
    # Drop unnecessary columns and join the data in one new dataframe
    train_non_linear_df = train_non_linear_df.drop(columns=columns_to_drop_linear, axis=1)
    train_time_domain_df = train_time_domain_df.drop(columns=columns_to_drop_time, axis=1)
    
    # Merge the two dataframes
    train_data = pd.merge(train_non_linear_df, train_time_domain_df, on=['uuid'], how='inner')
    train_data['HR'] = train_data['HR'].astype(int)
    # train_data['HR'] = data_pd['HR'].astype(int)
    train_data.to_csv(os.path.join(data_path, 'Heart_Rate_Prediction/train_data/train.csv'), index=False)
    print('The merged train data contains {} rows and {} columns.'.format(train_data.shape[0], train_data.shape[1]))
    
    # Simple Statistical Analysis
    mean_heart_rate = train_data['HR'].mean()
    median_heart_rate = train_data['HR'].median()
    std_heart_rate = train_data['HR'].std()
    
    print('The mean heart rate is {:.2f} bpm.'.format(mean_heart_rate))
    print('The median heart rate is {:.2f} bpm.'.format(median_heart_rate))
    print('The standard deviation of heart rate is {:.2f} bpm.'.format(std_heart_rate))
    
    
    # Test data pipeline
    test_non_linear_df  = pd.read_csv(test_non_linear_path)
    test_time_domain_df = pd.read_csv(test_time_domain_path)
    print('The initial non linear  test data contains {} rows and {} columns.'.format(test_non_linear_df.shape[0], test_non_linear_df.shape[1]))
    print('The initial time domain test data contains {} rows and {} columns.'.format(test_time_domain_df.shape[0], test_time_domain_df.shape[1]))
    
    # Drop unnecessary columns and join the data in one new dataframe
    test_non_linear_df = test_non_linear_df.drop(columns=columns_to_drop_linear, axis=1)
    test_time_domain_df = test_time_domain_df.drop(columns=columns_to_drop_time, axis=1)

    # Merge the two dataframes
    test_data = pd.merge(test_non_linear_df, test_time_domain_df, on=['uuid'], how='inner')
    test_data.to_csv(os.path.join(data_path, 'Heart_Rate_Prediction/test_data/test.csv'), index=False)
    print('The merged test data contains {} rows and {} columns.'.format(test_data.shape[0], test_data.shape[1]))


# --------------
# 3. Function to preprocess Stress-predict dataset
# --------------
def preprocess_stress_data(data_path):
    data = pd.read_csv(os.path.join(data_path, 'Stress_Predict/data.csv'))
    heart_rates = data['HR'].value_counts()
    labels = data['Label'].value_counts()
    
    # Simple Statistical Analysis
    mean_heart_rate = data['HR'].mean()
    median_heart_rate = data['HR'].median()
    std_heart_rate = data['HR'].std()
    
    print('The mean heart rate is {:.2f} bpm.'.format(mean_heart_rate))
    print('The median heart rate is {:.2f} bpm.'.format(median_heart_rate))
    print('The standard deviation of heart rate is {:.2f} bpm.'.format(std_heart_rate))
    
    # Create a new column for HRV, calculate the difference between consecutive heart rates every 15 seconds
    step = 15
    
    # New HRV column to be filled
    hrv_values = []

    for i in range(0, len(data), step):
        chunk = data.iloc[i:i+step]
        if len(chunk) == step:
            hr_list = chunk['HR'].tolist()
            hrv_result = compute_hrv_from_heart_rate(hr_list, use_jitter=True)
            hrv_rmssd = hrv_result[0]['rmssd']
            hrv_values.extend([hrv_rmssd] * 15)
        else:
            hrv_values.extend([np.nan] * len(chunk))

    # Assign to new column
    data['HRV'] = hrv_values

    # Export the data and delete downsampled data
    data = data[['Participant', 'HR', 'HRV', 'Label']]
    data.to_csv(os.path.join(data_path, 'Stress_Predict/hrv.csv'), index=False)


# --------------
# 4. Function to preprocess SWELL dataset
# --------------
def preprocess_swell_data(data_path):
    # Data path 
    swell_train_path= os.path.join(data_path,  'SWELL/final/train.csv')
    swell_test_path = os.path.join(data_path,  'SWELL/final/test.csv')
    
    # Load the SWELL data
    swell_train_df = pd.read_csv(swell_train_path)
    swell_test_df = pd.read_csv(swell_test_path)
    print(swell_train_df.head())
    
    columns = ['MEAN_RR', 'MEDIAN_RR', 'SDRR', 'SDRR_RMSSD', 'SDSD', 'pNN25', 'pNN50', 'SD1', 'SD2', 'KURT', 'SKEW', 
           'MEAN_REL_RR', 'MEDIAN_REL_RR', 'SDRR_REL_RR', 'RMSSD_REL_RR', 'SDSD_REL_RR', 'SDRR_RMSSD_REL_RR','KURT_REL_RR', 'SKEW_REL_RR', 
           'VLF', 'VLF_PCT', 'LF', 'LF_PCT', 'LF_NU', 'HF', 'HF_PCT', 'HF_NU', 'TP', 'LF_HF', 'HF_LF', 'sampen', 'higuci', 'datasetId']
    
    # Drop unnecessary columns and join the data in one new dataframe
    swell_train_df = swell_train_df.drop(columns=columns, axis=1)
    swell_test_df  = swell_test_df.drop(columns=columns, axis=1)
    
    # Train data pipeline
    swell_train_df['RMSSD'] = swell_train_df['RMSSD'].astype(int)
    swell_train_df['HR'] = swell_train_df['HR'].astype(int)
    swell_train_df.to_csv(os.path.join(data_path, 'SWELL/train.csv'), index=False)
    print('The SWELL train data contains {} rows and {} columns.'.format(swell_train_df.shape[0], swell_train_df.shape[1]))
    
    # Test data pipeline
    swell_test_df['RMSSD'] = swell_test_df['RMSSD'].astype(int)
    swell_test_df['HR'] = swell_test_df['HR'].astype(int)
    swell_test_df.to_csv(os.path.join(data_path, 'SWELL/test.csv'), index=False)
    print('The SWELL test data contains {} rows and {} columns.'.format(swell_test_df.shape[0], swell_test_df.shape[1]))

    # Simple Statistical Analysis
    mean_heart_rate = swell_train_df['HR'].mean()
    median_heart_rate = swell_train_df['HR'].median()
    std_heart_rate = swell_train_df['HR'].std()

    print('The mean heart rate is {:.2f} bpm.'.format(mean_heart_rate))
    print('The median heart rate is {:.2f} bpm.'.format(median_heart_rate))
    print('The standard deviation of heart rate is {:.2f} bpm.'.format(std_heart_rate))

def main():
    if len(sys.argv) < 3:
        print("Usage: python preprocess.py <data_path> <dataset_index>")
        sys.exit(1)

    data_path = sys.argv[1]
    try:
        dataset_index = int(sys.argv[2])
    except ValueError:
        print("Error: dataset_index must be an integer.")
        sys.exit(1)

    if not os.path.exists(data_path):
        print(f"Error: Data path '{data_path}' does not exist.")
        sys.exit(1)

    preprocessing_tasks = {
        0: ("Healthcare data", preprocess_healthcare_data),
        1: ("Heart Rate Prediction data", preprocess_heartrate_data),
        2: ("Stress Predict data", preprocess_stress_data),
        3: ("SWELL data", preprocess_swell_data),
    }

    if dataset_index in preprocessing_tasks:
        dataset_name, preprocessing_function = preprocessing_tasks[dataset_index]
        if preprocessing_function:
            print(f"Preprocessing {dataset_name}...")
            preprocessing_function(data_path)
            print("Preprocessing completed successfully.")
        else:
            print(f"Preprocessing for {dataset_name} is not yet implemented.")
    else:
        valid_indices = ", ".join(map(str, preprocessing_tasks.keys()))
        print(f"Error: Invalid dataset index. Please choose from: {valid_indices}.")
        sys.exit(1)

if __name__ == "__main__":
    main()
