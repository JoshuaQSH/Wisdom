import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Union, Dict


PREPARE_CASE = ['healthcare', 'hrp', 'swell', 'stresspred']


# -----------
# Function to compute HRV from heart rate data
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

    # Plot statistical data for the dataset
    hrvs = data_pd['HRV'].value_counts()
    nrows = 1
    ncols = 1

    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 8))

    axs.bar(hrvs.index, hrvs.values)
    axs.set_title('Heart Rate Variability Distribution')
    axs.set_xlabel('Heart Rate Variability')
    axs.set_ylabel('Frequency')
    
    fig.savefig(os.path.join(data_path, 'Healthcare/hrv_distribution.png'))
    plt.close(fig)

    # Detete the resampled data
    os.remove(os.path.join(data_path, 'Healthcare/resampled_data.csv'))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <data_path>")
        sys.exit(1)

    data_path = sys.argv[1]
    if not os.path.exists(data_path):
        print(f"Data path {data_path} does not exist.")
        sys.exit(1)

    preprocess_healthcare_data(data_path)
    print("Preprocessing completed successfully.")
