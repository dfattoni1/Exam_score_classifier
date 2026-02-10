import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

### create countplot
def make_countplot(data: list, var: str):
    """
    Function to make a count plot with title, ax labels, and bar 
    labels with the proportion for each category.
    
    :param data: List or Pandas Series containing the data with the categories
    :type data: list
    :param var: Name of the variable
    :type var: str
    """
    plt.figure()
    ax = sns.countplot(x = data, palette = "bright")

    for c in ax.containers:
        labels = ["{:.1%}".format(val.get_height() / len(data)) for val in c]
        ax.bar_label(c, labels = labels)

    plt.title("{} Class Distribution".format(var.title()), fontsize = 16)
    plt.xlabel("Categories")
    plt.ylabel("Frequency")
    plt.show()
    plt.clf()

### create histogram
def make_hist(data: list, var: str):
    """
    Function to make a histogram with title, ax labels, and vertical lines for
    the mean and median values of the data being passed.
    
    :param data: List or Pandas Series containing numerical data
    :type data: list
    :param var: Name of the variable
    :type var: str
    """
    plt.figure()
    sns.histplot(x = data, alpha = 0.5, color = "blue")
    plt.axvline(np.mean(data), color = "red", linestyle = "dashed", label = "Mean")
    plt.axvline(np.median(data), color = "green", linestyle = "dashed", label = "Median")
    plt.title("{} Distribution".format(var.title()))
    plt.xlabel(var.title())
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
    plt.clf()

### create scatter plot
def make_scatter(data_1: list, var_1: str, data_2: list, var_2: str, data_3 = None):
    """
    Function to make a scatter plot with title, ax labels, and hue (optional)
    
    :param data_1: List or Pandas Series containing numerical data for the first variable
    :type data_1: list
    :param var_1: Name of the first variable
    :type var_1: str
    :param data_2: List or Pandas Series containing numerical data for the second variable
    :type data_2: list
    :param var_2: Name of the second variable
    :type var_2: str
    :param data_3: List or Pandas Series containing categorical data for the hue of the plot
    """
    plt.figure()
    sns.scatterplot(x = data_1,
                    y = data_2,
                    hue = data_3,
                    color = "blue",
                    alpha = 0.5,
                    palette = "bright")
    plt.title("{} against {}".format(var_2.title(), var_1.title()))
    plt.xlabel(var_1.title())
    plt.ylabel(var_2.title())
    plt.show()
    plt.clf()

### create bar plot for bivariate analysis (categorical and numerical)
def make_biv_barplot(df, cat_var: str, num_var: str):
    """
    Function to create a bar plot for bivariate analysis for a categorical and a numerical variable.
    The function takes in the two names of the variables and gets the mean of the numerical variable
    for each category in the categorical variable. It then creates a bar plot with this.
    
    :param df: DataFrame containing the two variables.
    :type df: DataFrame
    :param cat_var: Column name of the categorical variable
    :type cat_var: str
    :param num_var: Column name of the numerical variable
    :type num_var: str
    """
    grouped_data = df.groupby(cat_var)[num_var].mean().reset_index()
    grouped_data_dict = {}

    for key, val in zip(grouped_data[cat_var], grouped_data[num_var]):
        grouped_data_dict[key] = val

    plt.figure()
    ax = sns.barplot(x = grouped_data_dict.keys(),
                     y = grouped_data_dict.values(),
                     palette = "bright")
    
    for c in ax.containers:
        ax.bar_label(c, fmt = "{:,.1f}")
        
    plt.title("Average for each category", fontsize = 16)
    plt.ylabel("Average {}".format(num_var), fontsize = 12)
    plt.xlabel(cat_var, fontsize = 12)
    plt.show()
    plt.clf()

    return grouped_data