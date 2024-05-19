import matplotlib.pyplot as plt
from matplotlib import colormaps
from pandas.plotting import scatter_matrix
import seaborn as sns
import streamlit as st


class DataVisualizer:
    """
    A class for visualizing data using various plots.
    """
    def __init__(self, data):
        """
        Initializes the DataVisualizer class with the provided data.

        Parameters:
        - data (DataFrame): The dataset to visualize.
        """
        self.data = data

    def scatter_plot(self, x_variable, y_variable, grid=False):
        """
        Create a scatter plot.

        Parameters:
        - x_variable (str): Name of the variable for the x-axis.
        - y_variable (str): Name of the variable for the y-axis.
        """
        fig, ax = plt.subplots()
        sns.scatterplot(x=x_variable, y=y_variable, data=self.data)
        return fig

    def histogram(self, variable, bins, grid=False):
        """
        Create a histogram.

        Parameters:
        - variable (str): Name of the variable for the histogram.
        """
        fig, ax = plt.subplots()
        sns.histplot(self.data[variable], bins = bins)
        return fig

    def box_plot(self, x_variable, y_variable, grid=False):
        """
        Create a box plot.

        Parameters:
        - x_variable (str): Name of the variable for the x-axis.
        - y_variable (str): Name of the variable for the y-axis.
        """
        fig, ax = plt.subplots()
        sns.boxplot(x=x_variable, y=y_variable, data=self.data)
        return fig

    def bar_plot(self, x_variable, y_variable, grid=False):
        """
        Create a bar plot.

        Parameters:
        - x_variable (str): Name of the variable for the x-axis.
        - y_variable (str): Name of the variable for the y-axis.
        """
        fig, ax = plt.subplots()
        sns.barplot(x=x_variable, y=y_variable, data=self.data)
        return fig

    def heatmap(self, heatmap_variables):
        """
        Create a heatmap.

        Parameters:
        - variables (list): List of variables for the heatmap.
        """
        heatmap_data = self.data[heatmap_variables].corr()
        fig, ax = plt.subplots()
        sns.heatmap(heatmap_data, annot=True, cmap="coolwarm")
        return fig
        

    def line_plot(self, x_variable, y_variable, scale=None, grid=False):
        """
        Create a line plot.

        Parameters:
        - x_variable (str): Name of the variable for the x-axis.
        - y_variable (str): Name of the variable for the y-axis.
        - scale (str or None): Scaling of the plot (e.g., "linear", "log", "symlog", "logit").
        - grid (bool): Whether to display grid lines.
        """
        fig, ax = plt.subplots()
        if scale:
            ax.set_yscale(scale)
        sns.lineplot(x=x_variable, y=y_variable, data=self.data)
        return fig


    def pairplot(self, variables, diagonal="scatter"):
        """
        Create a pairplot.

        Parameters:
        - variables (list): List of variables for the pairplot.
        - diagonal (str): Type of plot for the diagonal subplots. Defaults to "scatter".
        """
        axes = scatter_matrix(self.data[variables], diagonal=diagonal, edgecolor="black", alpha=.7)
        return plt.gcf()

    @staticmethod
    def covariance_matrix(data):
        """
        Create a covariance matrix heatmap.

        Parameters:
        - data (DataFrame): The dataset to compute the covariance matrix and visualize.
        """
        cmap = colormaps.get_cmap("twilight")
        fig, ax = plt.subplots()
        sns.heatmap(data.cov(numeric_only=True), annot=True)
        return fig


class ModelEvaluation:
    """
    A class for evaluating machine learning models.
    """
    @staticmethod
    def confusion_matrix(confusion_matrix, ticks, labels):
        """
        Create a confusion matrix plot.

        Parameters:
        - confusion_matrix (array): Confusion matrix.
        - ticks (array): Array of tick locations.
        - labels (array): Array of label names.
        """
        fig, ax = plt.subplots()
        rotation = len(labels)//5 * 15
        cmap = colormaps.get_cmap("twilight")
        
        sns.heatmap(confusion_matrix,  
                    annot=True,
                    linewidths=.1, 
                    ax=ax)
        ax.set_xticks(ticks=ticks, 
                      labels=labels, 
                      rotation=min(rotation,90))
        ax.set_yticks(ticks=ticks, 
                      labels=labels, 
                      rotation=0)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Target label")
        return fig
        
    @staticmethod
    def classification_report(precision_recall_fscore_support, y_ticks, y_labels):
        """
        Create a classification report plot.

        Parameters:
        - precision_recall_fscore_support (array): Precision, recall, fscore, and support values.
        - y_ticks (array): Array of tick locations.
        - y_labels (array): Array of label names.
        """
        fig, ax = plt.subplots()
        
        x_labels=["Precision", "Recall", "F1 score", "Support"]
        cmap = colormaps.get_cmap("twilight")
        
        
        sns.heatmap(precision_recall_fscore_support,
                    annot=True,linewidths=.1, 
                    fmt='.3g', cmap=cmap, 
                    vmin=0, vmax=1, ax=ax)
        
        ax.set_xticklabels(x_labels, 
                           rotation=0)
        ax.set_yticks(ticks=y_ticks, 
                      labels=y_labels, 
                      rotation=0)
        ax.set_title("Classification report")
        ax.set_xlabel("Metrics")
        ax.set_ylabel("Target Class")
        return fig
    

    @staticmethod
    def hlines(features, feature_names):
        """
        Create a horizontal line plot for feature importance.

        Parameters:
        - features (DataFrame): Features and their importance values.
        - feature_names (array): Array of feature names.
        """
        fig, ax = plt.subplots()
        y = range(len(features),0,-1)
        
        
        plt.yticks(y, features[0])
        plt.ylabel("Features")
        plt.xlabel("Feature Importance")
        plt.title("Feature Importance")
        plt.ylim(.5,len(features)+.5)
    
        ax.hlines(y, [0]*len(features), features[1], color="blue", alpha=.5)
        ax.scatter(features[1], y, color="blue", alpha=1)
        return fig

    @staticmethod
    def bar(comparison):
        """
        Create a bar plot for model comparison.

        Parameters:
        - comparison (DataFrame): DataFrame containing model comparison data.
        """
        fig, ax = plt.subplots()
        comparison.iloc[:,1:].T.plot(kind="barh", ax=ax)
        if (comparison.iloc[:,1:].max(axis=1) - comparison.iloc[:,1:].min(axis=1)/comparison.iloc[:,1:].min(axis=1))\
        .max() >= 50:
            ax.set_xscale("log")
        for lg in plt.legend().legendHandles:
            lg.set_alpha(.5)
        return fig

