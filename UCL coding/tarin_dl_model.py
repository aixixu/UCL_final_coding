import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_curve, f1_score, precision_score, recall_score, confusion_matrix, cohen_kappa_score, hamming_loss, roc_curve
import src.utils
src.utils.gpu_setup()
from omegaconf import OmegaConf
conf = OmegaConf.load("./../podcast-dataset/config.yaml")

def crea_binary_deep_learning_model(feature_num, target_num):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[feature_num,]))
    model.add(keras.layers.Dense(500, activation = "relu"))
    model.add(keras.layers.Dense(300, activation = "relu"))
    model.add(keras.layers.Dense(100, activation = "relu"))
    model.add(keras.layers.Dense(target_num, activation = "sigmoid"))
    return model

def crea_multi_deep_learning_model(feature_num, target_num):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[feature_num,]))
    model.add(keras.layers.Dense(500, activation = "relu"))
    model.add(keras.layers.Dense(300, activation = "relu"))
    model.add(keras.layers.Dense(100, activation = "relu"))
    model.add(keras.layers.Dense(target_num, activation = "softmax"))
    return model

def metrics_dl_model_binary(y_test, y_predict):
    acc = accuracy_score(y_test, y_predict)
    f1 = f1_score(y_test, y_predict)
    pr = precision_score(y_test, y_predict)
    re = recall_score(y_test, y_predict)
    return acc, f1, pr, re

def metrics_dl_model_multi(y_test, y_predict):
    f1 = f1_score(y_test, y_predict, average='macro')
    pr = precision_score(y_test, y_predict, average='macro')
    re = recall_score(y_test, y_predict, average='macro')
    kappa = cohen_kappa_score(y_test, y_predict)
    return f1, pr, re, kappa
        
    
    