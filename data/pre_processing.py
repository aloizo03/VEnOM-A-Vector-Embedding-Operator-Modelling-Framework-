import argparse
import os
import numpy as np
import pandas as pd


def pre_process_adult_dataset(df):

    race_unique = np.unique(df['race'].tolist())
    gender_unique = np.unique(df['gender'].tolist())
    income_unique = np.unique(df['income'].tolist())

    race_to_idx = {race: count for count, race in enumerate(race_unique)}
    gender_to_idx = {gender: count for count, gender in enumerate(race_unique)}
    income_to_idx = {income: count for count, income in enumerate(income_unique)}

    df["race"] = df["race"].map(race_to_idx)
    df["gender"] = df["gender"].map(gender_to_idx)
    df["income"] = df["income"].map(income_to_idx)

    return df


def pre_processing_wine_dataset(df):
    color_unique = np.unique(df['color'].tolist())
    color_to_idx = {color: count for count, color in enumerate(color_unique)}
    df['color'] = df["color"].map(color_to_idx)
    df = df.fillna(-99)
    return df


def save_df(out_path, filename, df):
    global_path = os.getcwd()
    out_path = os.path.join(global_path, out_path)
    out_path = os.path.join(out_path, f'{filename}.csv')
    df.to_csv(out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Input datasets path")
    parser.add_argument("-o", "--out", type=str, help="Output path of the process dataset")
    parser.add_argument("-t", "--dataset-name", type=str, help="What dataset name is ? (adult, hpw)")
    args = parser.parse_args()
    input_data_path = args.input
    output_data_path = args.out
    dataset_name = args.dataset_name

    df = pd.read_csv(input_data_path)
    if dataset_name.lower() == 'adult':
        df_process = pre_process_adult_dataset(df)
    elif dataset_name.lower() == 'wine_quality':
        df_process = pre_processing_wine_dataset(df)
    else:
        AssertionError(f'The input dataset are not exist!!!')

    save_df(out_path=output_data_path, filename=dataset_name.lower(), df=df_process)

