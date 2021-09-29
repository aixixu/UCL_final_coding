import os
import math
from pathlib import Path
import IPython
import pandas as pd
import numpy as np

from src.data import load_metadata, find_paths

# Read csv file
def load_csv_data(data_path, sep=','):
    return pd.read_csv(data_path,delimiter=sep)

# Find the file path and timestamp by uri
def uri_find_path(uri,path,file_type, metadata):
    episode_uri, start_temestamp = uri.split("_")
    episode = metadata[(metadata.episode_uri == episode_uri).values]
    files = find_paths(episode, path, file_type)
    return files[0], int(float(start_temestamp))

# Find the feature path and start timestamp of the labeled raw file through the raw path
def feature_paths_starttimestamp_labeled(segments,rawpath,appendix,metadata):
    paths = []
    starts = []
    for segment in segments['uri']:
        path, start = uri_find_path(segment,rawpath,appendix,metadata)
        paths.append(path)
        starts.append(start)
    return paths, starts

# get the feature data from the feature path and feature starttimestamp
def get_feature_data(feature_path, feature_starttimestamp):
    feature_labeled = []
    feature_labeled_index=[]
    i=0
    for path, start in zip(feature_path, feature_starttimestamp):
        try:
            feature_labeled.append(pd.read_hdf(path).iloc[start,:])
            feature_labeled_index.append(i)
            i+=1
        except:
            continue
    feature_df = pd.DataFrame(pd.concat(feature_labeled, axis=1, ignore_index=True).values.T, index=feature_labeled_index, columns=pd.concat(feature_labeled, axis=1, ignore_index=True).index)
    return feature_df

# drop some features
def drop_features(feature_df,features_to_be_removed):
    return feature_df.drop(features_to_be_removed, axis=1)

# select some features
def select_features(feature_df,features_to_be_selected):
    return feature_df[features_to_be_selected]

# get three class by input class name without null value
# we have three class entertaining subjective and discussion
# every class have a columns on raw data
def get_filter_label(labeled_dataframe, class_name):
    labeled_list_filter =  [elem for elem in labeled_dataframe[class_name].tolist() if elem != "['']" and elem != '[]' and elem != "['NA']"]
    return labeled_list_filter

# get three class by input class name
def get_label(labeled_dataframe, class_name):
    labeled_list = labeled_dataframe[class_name].tolist()
    return labeled_list