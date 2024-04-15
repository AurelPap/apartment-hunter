###########
# Library #
###########

import math
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


#############
# Functions #
#############

# Some visualization functions for quick view

def boxplot_whiskers(data):
    Q3 = data.quantile(0.75)
    Q1 = data.quantile(0.25)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    if lower_bound < 0:
        lower_bound = data.min()
    return lower_bound, upper_bound

def boxplot_whiskers_iterative_outliers_removal(dataframe, column_name):
    # 1. Creating a df `data_no_outliers` with initial removal of outliers
    dataframe = dataframe.copy()
    column = column_name
    data_whiskers = boxplot_whiskers(dataframe[column])
    data_no_outliers = dataframe[(dataframe[column] >= data_whiskers[0]) & (dataframe[column] <= data_whiskers[1])]

    # 2. Calculating the whisker bounds in the trimmed df `data_no_outliers`
    sqm_living_no_outliers_whiskers = boxplot_whiskers(data_no_outliers[column])

    # 3. Reiterating the removal of outliers until the whisker bounds remain constant
    previous_whiskers = sqm_living_no_outliers_whiskers
    while True:
        data_no_outliers = data_no_outliers[(data_no_outliers[column] >= boxplot_whiskers(data_no_outliers[column])[0]) & (data_no_outliers[column] <= boxplot_whiskers(data_no_outliers[column])[1])]
        current_whiskers = boxplot_whiskers(data_no_outliers[column])
        if previous_whiskers == current_whiskers:
            break
        previous_whiskers = current_whiskers
    return current_whiskers


def boxplot(data, label, title='', axis='y', figsize=(6,6)):
    plt.figure(figsize=figsize)
    sns.boxplot(x=data) if axis == 'x' else sns.boxplot(y=data)
    if title != '':
        title = f'{title} '
    title = f'{title}Boxplot of {label}'
    plt.title(title)
    plt.xlabel(label) if axis == 'x' else plt.ylabel(label)
    plt.ticklabel_format(style='plain', axis=axis)
    plt.show()

def boxplot_categories(dataframe, column_name_x_axis, column_name_y_axis, figsize=(6, 6), xticksrotation=0, order_by_median_flag=False, title=None):
    if order_by_median_flag == False:
        pass
    else:
        try:
            median = dataframe.groupby(column_name_x_axis)[column_name_y_axis].median().sort_values()
        except:
            median = dataframe.groupby(column_name_y_axis)[column_name_x_axis].median().sort_values()
        ordered = median.index
    plt.figure(figsize=figsize)
    if order_by_median_flag == False:
        sns.boxplot(x=column_name_x_axis, y=column_name_y_axis, data=dataframe)
    else:
        sns.boxplot(x=column_name_x_axis, y=column_name_y_axis, data=dataframe, order=ordered)
    if title == None:
        plt.title(f'Box Plot of {column_name_y_axis} by {column_name_x_axis}')
    else:
        plt.title(title)
    plt.xlabel(column_name_x_axis)
    plt.xticks(rotation=xticksrotation)
    plt.ylabel(column_name_y_axis)
    if column_name_y_axis == 'price':
        plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "${:,}".format(int(x))))
    elif column_name_x_axis == 'price':
        plt.gca().get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "${:,}".format(int(x))))
    plt.grid(True)
    plt.show()

def histogram(data, bins, label, title='', figsize=(8, 6), xticksrotation=0, skewness=(0, 0), kurtosis=(0, 0), mean=False, median=False, legend=False):
    plt.figure(figsize=figsize)
    plt.hist(data, bins=bins, color='skyblue', edgecolor='black')
    if mean == True:
        mean_price = data.mean()
        plt.axvline(mean_price, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_price:.2f}')
    if median == True:
        median_price = data.median()
        plt.axvline(median_price, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_price:.2f}')
    if skewness != (0, 0):
        skewness_value = data.skew()
        plt.text(skewness[0], skewness[1], f'Skewness: {skewness_value:.2f}', transform=plt.gca().transAxes)
    if kurtosis != (0, 0):
        kurtosis_value = data.kurtosis()
        plt.text(kurtosis[0], kurtosis[1], f'Kurtosis: {kurtosis_value:.2f}', transform=plt.gca().transAxes)
    plt.xlabel(label)
    plt.ylabel('Frequency')
    if legend == True:
        plt.legend()
    if title != '':
        title = f'{title} '
    plt.title(f'{title}{label} data distribution')
    plt.xticks(rotation=xticksrotation)
    plt.ticklabel_format(style='plain', axis='x')
    plt.show()

def barplot(value_counts, title, xlabel, ylabel, figsize=(8, 6), xticksrotation=0):
    plt.figure(figsize=figsize)
    sns.barplot(x=value_counts.index, y=value_counts.values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=xticksrotation)
    plt.grid(True)
    plt.show()
    
    
# Grouping continuous values by histogram bins to a new column:
    
def create_column_histogram_bin_number(dataframe, column, bins, new_column_name):
    dataframe[new_column_name] = np.digitize(dataframe[column], bins=np.arange(dataframe[column].min(), dataframe[column].max(), (dataframe[column].max()-dataframe[column].min())/bins))
    return dataframe

# Calculating histogram bin edges (for quick view):

def histogram_bins_values(data, bins, print_flag=None):
    bin_size = (data.max() - data.min()) / bins
    bin_start = data.min()
    bin_values = []
    for bin in range(bins):
        bin_start = round(bin_start, 2)
        bin_end = round(bin_start + bin_size, 2)
        if print_flag != None:
            print(f'Bin {bin + 1}:  starts at {bin_start} ends at {bin_end}')
        bin_values.append((bin_start, bin_end))
        bin_start = bin_end
    return bin_values

'''We will adjust pandas display option **to handle scientific notation** in the *describe* and other functions:
  
- `pd.set_option('display.float_format', lambda x: '%.2f' % x)` sets the display format for floating-point numbers to two decimal places.  
  
- `pd.reset_option('display.float_format')` resets the display format option.'''

pd.set_option('display.float_format', lambda x: '%.2f' % x)


# Nomination des variables qualitatives et quantitatives
variables_quantitatives = [
    "price",
    "bedrooms",
    "bathrooms",
    "sqm_living",
    "sqm_lot",
    "floors",
    "sqm_above",
    "sqm_basement",
    "yr_built",
    "yr_renovated",
    "sqm_living15",
    "sqm_lot15"
]

colors_quantitative = [
    '#FF6F61',  # Rouge
    '#6B5B95',  # Violet
    '#88B04B',  # Vert
    '#F7CAC9',  # Rose
    '#92A8D1',  # Bleu clair
    '#955251',  # Brun
    '#B565A7',  # Violet clair
    '#009B77',  # Vert clair
    '#DAA520',  # Or
    '#1E90FF',  # Bleu dodger
    '#FFA500',  # Orange
    '#800000'   # Rouge foncé
]


variables_qualitatives = [
    "waterfront",
    "view",
    "condition",
    "grade",
]


colors_qualitative = [
    'midnightblue',
    'orange',
    'green',
    'purple'
]