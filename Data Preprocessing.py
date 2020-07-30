import pydicom
import pandas as pd
import numpy as np
import os
import glob
import scipy.ndimage
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
train_csv_path = 'Data/train.csv'
test_csv_path = 'Data/test.csv'
train_dicom_path = 'Data/train/'


def csv_to_df(csv_path):
    df = pd.read_csv(csv_path)
    return df


"""def csv_category_encoding(csv_df):
    ohe_smoking = OneHotEncoder()
    ohe_sex = OneHotEncoder()
    csv_df['SmokingStatus'] = ohe_smoking.fit_transform(csv_df['SmokingStatus'])
    csv_df['Sex'] = ohe_sex.fit_transform(csv_df['Sex'])
    return csv_df"""


def csv_get_dummies(csv_df):
    columns = ['Sex', 'SmokingStatus']
    csv_df = pd.get_dummies(csv_df, columns=columns)
    return csv_df


def csv_scaling(csv_df):
    ss_fvc = StandardScaler(with_mean=False)
    # ct = ColumnTransformer(['somename', ss_fvc, ['FVC']])
    csv_df['FVC'] = ss_fvc.fit_transform(csv_df[['FVC']])
    csv_df['Percent'] = csv_df['Percent']/100
    return csv_df






"""test_df = csv_to_df(test_csv_path)
test_df = csv_get_dummies(test_df)
test_df = csv_scaling(test_df)"""


