import pandas as pd
import numpy as np
import os
from surprise import Reader, Dataset, SVD, model_selection
from sklearn.metrics import mean_squared_error
from math import sqrt
import time


def read_data(file_path):
    # read training_set.txt to dataframe
    # skip date
    sub_df = pd.read_csv(file_path, header = None, names = ['Cust_Id', 'Rating'], usecols = [0, 1])
    sub_df['Rating'] = sub_df['Rating'].astype(float)
    return sub_df


def merge_data(file_dir):

    # merge all the training_set.txt
    # num can control the numbers of txt files
    num = 0
    for file_name in sorted(os.listdir(file_dir)):

        file_path = os.path.join(file_dir, file_name)
        num += 1

        if num == 1:
            df = read_data(file_path)
        elif num > 1 and num <=2:
            sub_df = read_data(file_path)
            df = df.append(sub_df)

            if num % 50 == 0:
                print('the {} file'.format(num))

        else:
            break


    return df


def data_processing(df):

    # find the index of the line of movieID
    df_nan = pd.DataFrame(pd.isnull(df.Rating))
    df_nan = df_nan[df_nan['Rating'] == True]
    df_nan = df_nan.reset_index()

    movie_np = []

    # for set value of movieID, it's in ascending order,
    movie_id = 1

    # find the length of each movieID
    for i,j in zip(df_nan['index'][1:],df_nan['index'][:-1]):
        # numpy approach
        temp = np.full((1,i-j-1), movie_id)
        movie_np = np.append(movie_np, temp)
        movie_id += 1

    # Account for last record and corresponding length
    # numpy approach
    last_record = np.full((1,len(df) - df_nan.iloc[-1, 0] - 1),movie_id)
    movie_np = np.append(movie_np, last_record)

    # remove those Movie ID rows, (eg. 1: 2:) not use
    df = df[pd.notnull(df['Rating'])]

    df['Movie_Id'] = movie_np.astype(int)
    df['Cust_Id'] = df['Cust_Id'].astype(int)


    ###################### data slicing ######################
    f = ['count', 'mean']

    # Remove movie with too less reviews
    df_movie_summary = df.groupby('Movie_Id')['Rating'].agg(f)
    df_movie_summary.index = df_movie_summary.index.map(int)
    movie_benchmark = round(df_movie_summary['count'].quantile(0.5),0)
    drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

    # print('Movie minimum times of review: {}'.format(movie_benchmark))

    # Remove customer who give too less reviews
    df_cust_summary = df.groupby('Cust_Id')['Rating'].agg(f)
    df_cust_summary.index = df_cust_summary.index.map(int)
    cust_benchmark = round(df_cust_summary['count'].quantile(0.5),0)
    drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index

    # print('Customer minimum times of review: {}'.format(cust_benchmark))

    # print('Original Shape: {}'.format(df.shape))

    df = df[~df['Movie_Id'].isin(drop_movie_list)]
    df = df[~df['Cust_Id'].isin(drop_cust_list)]

    print('After Trim Shape: {}'.format(df.shape))

    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)



    return df


def train_test(df):

    ###################### train ######################
    reader = Reader()
    svd = SVD()

    border_line = int(df.shape[0] / 5)

    data = Dataset.load_from_df(df[['Cust_Id', 'Movie_Id', 'Rating']][:-border_line], reader)

    # train 1: cross_validate
    # model_selection.cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

    # train 2: not cross_validate
    trainset = data.build_full_trainset()
    svd.fit(trainset)

    ###################### test ######################
    test_df = df.iloc[-border_line:]

    print('test_df Shape: {}'.format(test_df.shape))
    data_matrix = np.array(test_df, dtype=np.int)

    Estimate_Score = []
    for user in data_matrix:
        Score = svd.predict(user[0], user[2]).est
        Estimate_Score.append(Score)

    loss = RMSE(Estimate_Score, data_matrix[:, 1])

    return loss


def RMSE(prediction, ground_truth):
    return sqrt(mean_squared_error(prediction, ground_truth))


if __name__ == '__main__':
    file_dir = '/home/zwj/Desktop/recommend/download/training_set/'

    read_data_start = time.time()
    df = merge_data(file_dir)
    df.index = np.arange(0,len(df))

    data_process_start = time.time()
    df = data_processing(df)

    rmse = train_test(df)

    total_time = time.time() - read_data_start
    print('total_time is : {}'.format(total_time))

    print('RMSE is : {}'.format(rmse))
