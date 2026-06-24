# File path configuration
BASE_PATH = rf"F:\Uni\Coding\PVS 1\\"
PATHS = {
    "labels": BASE_PATH + "dataset_labels.csv",
    "gps_mpu_left": BASE_PATH + "dataset_gps_mpu_left.csv",
    "gps_mpu_right": BASE_PATH + "dataset_gps_mpu_right.csv",
    "mpu_left": BASE_PATH + "dataset_mpu_left.csv",
    "mpu_right": BASE_PATH + "dataset_mpu_right.csv",
}


#test
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import preprocessing

def load_labels(path):
    return pd.read_csv(path)

def convert_labels_to_features(df_labels):
    def one_hot_to_label(df_in, classes, df_out, class_name):
        conditions = [df_in[col] == 1 for col in classes]
        choices = classes
        df_out[class_name] = np.select(conditions, choices, default='unknown')
        return df_out

    labels = pd.DataFrame()
    labels = one_hot_to_label(df_labels, ['dirt_road', 'cobblestone_road', 'asphalt_road'], labels, 'road')
    labels = one_hot_to_label(df_labels, ['paved_road', 'unpaved_road'], labels, 'condition')
    labels = one_hot_to_label(df_labels, ['no_speed_bump', 'speed_bump_asphalt', 'speed_bump_cobblestone'], labels, 'bumps')
    labels = one_hot_to_label(df_labels, ['good_road_right', 'regular_road_right', 'bad_road_right'], labels, 'quality_right')
    labels = one_hot_to_label(df_labels, ['good_road_left', 'regular_road_left', 'bad_road_left'], labels, 'quality_left')

    labels = labels.replace({
        'quality_right': { 'good_road_right': 2, 'regular_road_right': 1, 'bad_road_right': 0 },
        'quality_left':  { 'good_road_left': 2, 'regular_road_left': 1, 'bad_road_left': 0 },
    })
    labels['quality'] = labels[['quality_right', 'quality_left']].mean(axis=1)
    return labels.drop(columns=['quality_right', 'quality_left'])

def load_gps_mpu_data(path_left, path_right):
    return pd.read_csv(path_left), pd.read_csv(path_right)

def absolute_sensor_average(df_left, df_right, columns):
    acc_left = df_left[columns].abs().sum(axis=1) / len(columns)
    acc_right = df_right[columns].abs().sum(axis=1) / len(columns)
    return pd.concat([acc_left, acc_right], axis=1).mean(axis=1)

def compute_sensor_magnitudes(df_left, df_right):
    acc_cols = [col for col in df_left.columns if 'acc_' in col]
    gyro_cols = [col for col in df_left.columns if 'gyro_' in col]
    mag_cols = [col for col in df_left.columns if 'mag_' in col]

    mpu = pd.DataFrame()
    mpu['acceleration'] = absolute_sensor_average(df_left, df_right, acc_cols)
    mpu['gyro'] = absolute_sensor_average(df_left, df_right, gyro_cols)
    mpu['mag'] = absolute_sensor_average(df_left, df_right, mag_cols)
    return mpu

def generate_combined_dataset():
    df_labels = load_labels(PATHS['labels'])
    labels = convert_labels_to_features(df_labels)
    df_left, df_right = load_gps_mpu_data(PATHS['gps_mpu_left'], PATHS['gps_mpu_right'])
    mpu = compute_sensor_magnitudes(pd.read_csv(PATHS['mpu_left']), pd.read_csv(PATHS['mpu_right']))
    gps = df_left[['timestamp_gps', 'latitude', 'longitude', 'speed']]
    return pd.concat([gps, labels, mpu], axis=1)

def save_dataset(df, filename="combined_pvs1.csv"):
    df.to_csv(filename, index=False)

def run_kmeans_clustering(df, features=['speed', 'quality', 'acceleration', 'gyro', 'mag'], n_clusters=3):
    scaler = preprocessing.MaxAbsScaler()
    data_scaled = scaler.fit_transform(df[features])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    df['cluster'] = kmeans.fit_predict(data_scaled)
    return df

def plot_label_distributions(df_labels):
    df_labels.hist(figsize=(15, 10))
    plt.tight_layout()
    #plt.show()
    plt.savefig("label distribution.png")

def plot_temperature(df):
    plt.plot(df['timestamp'], df['temp_dashboard'], label='dashboard')
    plt.plot(df['timestamp'], df['temp_above_suspension'], label='above')
    plt.plot(df['timestamp'], df['temp_below_suspension'], label='below')
    plt.legend()
    #plt.show()
    plt.savefig("temperature.png")

def plot_acceleration_gyro_mag(ts, acc, gyro, mag):
    fig, ax = plt.subplots(3, 1, figsize=(12, 8))
    sns.lineplot(x=ts, y=acc, ax=ax[0]); ax[0].set_title("Acceleration")
    sns.lineplot(x=ts, y=gyro, ax=ax[1]); ax[1].set_title("Gyroscope")
    sns.lineplot(x=ts, y=mag, ax=ax[2]); ax[2].set_title("Magnetometer")
    plt.tight_layout()
    #plt.show()
    plt.savefig("acc_gyro_mag.png")

def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12,10))
    sns.heatmap(numeric_df.corr(), annot=True)
    #plt.show()
    plt.savefig("correlation_heatmap.png")

def plot_cluster_scatter(df, x, y):
    sns.scatterplot(x=x, y=y, hue='cluster', data=df, palette='hls')
    plt.title(f'Clustered by {x} vs {y}')
    #plt.show()
    plt.savefig("cluster_scatter.png")


if __name__=="__main__":

    df = generate_combined_dataset()
    save_dataset(df)
    df = run_kmeans_clustering(df)
    #plot_label_distributions(df)
    df_mpu = pd.read_csv(PATHS['mpu_left'])  # or mpu_right
    #plot_temperature(df_mpu)
    #plot_acceleration_gyro_mag(df['timestamp_gps'], df['acceleration'], df['gyro'], df['mag'])
    #plot_correlation_heatmap(df)
    plot_cluster_scatter(df, 'speed', 'quality')

