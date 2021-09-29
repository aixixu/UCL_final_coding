import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from scipy.stats import pearsonr


from load_data import *
from plot_feature_model import *

# variance filter return boolean columns value
def variance_filter(input_df,threshold):
    var_selected=VarianceThreshold(threshold=threshold)
    data_var_selected=var_selected.fit_transform(input_df)
    columns_variance_selected=var_selected.get_support(indices=False)
    return columns_variance_selected

# caculate the mean of the relevant features    
def get_mean_related_feature(input_df, old_features_list):
    value_store=[]
    for old_features in old_features_list:
        value_store.append(input_df[old_features])
    return np.mean(value_store, 0)

# pearsonr filter
def pearsonr_feature_filter(input_df, label, best_feature_num):
    pearsonr_selecter = SelectKBest(lambda X, Y: tuple(map(tuple,np.array(list(map(lambda x:pearsonr(x, Y), X.T))).T)), k=best_feature_num)
    data_pearsonr_selected = pearsonr_selecter.fit_transform(input_df, label)
    return pearsonr_selecter.get_support(indices=False)
    
# chi filter
def chi_feature_filter(input_df, label, best_feature_num):
    chi_selecter = SelectKBest(chi2, k=best_feature_num)
    data_chi_selected =  chi_selecter.fit_transform(input_df, label)
    return chi_selecter.get_support(indices=False)

# Feature deduplication
def get_distinct_feature(feature_df_1, feature_df_2):
    return set(feature_df_1.columns.tolist() + feature_df_2.columns.tolist())

# GBDT feature selection
def gbdt_feature_selection(input_df, label_df):
    GBDT_selecter = SelectFromModel(GradientBoostingClassifier())
    GBDT_selected_subjective_data = GBDT_selecter.fit_transform(yamnet_feature_labeled_df, yamnet_feature_labeled_df_sujective.label)
    GBDT_selected_subjective_columns = GBDT_selecter.get_support(indices=False)
    return GBDT_selected_subjective_columns

# GBDT feature selection
def gbdt_feature_selection(input_df, label_df):
    GBDT_selecter = SelectFromModel(GradientBoostingClassifier())
    GBDT_selected_subjective_data = GBDT_selecter.fit_transform(input_df, label_df)
    GBDT_selected_subjective_columns = GBDT_selecter.get_support(indices=False)
    return GBDT_selected_subjective_columns

# RFE feature selection
def rfe_feature_selection(estimator, min_features_num, input_df, label_df):
    if estimator == 'SVM':
        RFE_selecter = RFECV(estimator=LinearSVC(), min_features_to_select=min_features_num)
        RFE_selected_data = RFE_selecter.fit_transform(input_df, label_df)
        RFE_selected_columns = RFE_selecter.get_support(indices=False)
    elif estimator== 'LR':
        RFE_selecter = RFECV(estimator=LogisticRegression(), min_features_to_select=min_features_num)
        RFE_selected_data = RFE_selecter.fit_transform(input_df, label_df)
        RFE_selected_columns = RFE_selecter.get_support(indices=False)
    else:
        raise ValueError("Incorrect data format, please input 'SVM' or 'LR'")
    return RFE_selected_columns

