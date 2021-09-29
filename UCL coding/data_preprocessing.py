import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from load_data import *
from plot_feature_model import *

# add final class label
# final means it will be used as input for model training
def add_final_class_label(input_df, label_df):
    final_df = input_df.copy()
    final_df['label'] = pd.Series(label_df)
    return final_df

# add class label 1 and -1 not final label
def add_class_label(input_df, label_df, classname):
    tmp_df = input_df.copy()
    tmp_df['label']=-1
    class_lable_list = get_label(label_df, classname)
    if(classname == 'subjective'):
        for index, row in tmp_df.iterrows():
            if 'Approval' in class_lable_list[index] or 'Disapproval' in class_lable_list[index]  :
                tmp_df.loc[index,'label']=1
    elif(classname == 'discussion'):
        for index, row in tmp_df.iterrows():
            if 'conversation' in class_lable_list[index]:
                tmp_df.loc[index,'label']=1
    elif(classname == 'entertaining'):
        for index, row in tmp_df.iterrows():
            if 'storytelling' in class_lable_list[index]:
                tmp_df.loc[index,'label']=1
    return tmp_df

# min max scaler
def min_max_scaler(input_data):
    minmaxscaler_filter = MinMaxScaler().fit_transform(input_data)
    minmaxscaler_filter_df = pd.DataFrame(minmaxscaler_filter, columns = input_data.columns, index = input_data.index)
    return minmaxscaler_filter_df

# standar scaler
def standar_scaler(input_data):
    StandardScaler_yamnet = StandardScaler().fit_transform(input_data)
    StandardScaler_yamnet_df = pd.DataFrame(StandardScaler_yamnet, columns=input_data.columns, index = input_data.index)
    return StandardScaler_yamnet_df

# binary-class add label
def binary_class_add_label(labe_list, postive_content):
    labe_list_binary = []
    for label in labe_list:
        if postive_content in label:
            labe_list_binary.append(1)
        else:
            labe_list_binary.append(0)
    return labe_list_binary

# multi class add label
# remove the extra punctuation and keep only the numbers
def multi_class_add_label(input_label):
    input_label_transfer = []
    for elem in input_label:
        if elem != "['']" and elem != '[]' and elem != "['NA']":
            input_label_transfer.append(elem)
        else:
            input_label_transfer.append("")

    input_label_transfer2 = []
    for elem in input_label_transfer:
        input_label_transfer2.append(elem.replace("[","").replace("'","").replace("]",""))
    return input_label_transfer2

# raw minmax standar feature, not plot
def raw_minmax_stand(input_df, label_list):
    data_raw = input_df
    data_minmax = min_max_scaler(data_raw)
    data_minmax_standar = standar_scaler(data_minmax)

    return data_raw,data_minmax,data_minmax_standar

# split train set and test set
def split_train_test_set(input_df, input_label, split_size):
    X_train, X_test, y_train, y_test = train_test_split(input_df, input_label, test_size=split_size, random_state=123)
    return X_train, X_test, y_train, y_test


