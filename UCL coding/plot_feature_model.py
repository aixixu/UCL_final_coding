import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

from data_preprocessing import *
from feature_selection import *

# plot correlation heatmap, threshold
# If the correlation is below the threshold, don’t show
def plot_corr(input_df, threshold,save_path):
    corr_df = input_df.corr()
    corr_df[abs(corr_df) <= 0.80] = np.nan
    plt.figure(figsize=(25, 25))
    ax = sns.heatmap(
        corr_df, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=90,
        horizontalalignment='right'
    )
    plt.savefig(save_path, bbox_inches='tight')

# plot data
def plot_data(input_df, save_path):
    fig = sns.pairplot(input_df,hue='label')
    fig.savefig(save_path, bbox_inches='tight')

# plot raw minmax standar feature
def plot_raw_minmax_stand(input_df, label_list,savename):
    save_raw_path = savename + '_raw.png'
    save_minmax_path = savename + '_minmax.png'
    save_minmax_stand_path = savename + '_minmax_stand.png'
    
    data_raw = input_df
    plot_data(add_final_class_label(data_raw, label_list), save_raw_path)

    data_minmax = min_max_scaler(data_raw)
    plot_data(add_final_class_label(data_minmax, label_list), save_minmax_path)

    data_minmax_standar = standar_scaler(data_minmax)
    plot_data(add_final_class_label(data_minmax_standar, label_list), save_minmax_stand_path)
    return data_raw,data_minmax,data_minmax_standar

