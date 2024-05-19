## Import necessary libraries and modules
from category_encoders.one_hot import OneHotEncoder
from category_encoders.ordinal import OrdinalEncoder
from joblib import dump # For saving models
import numpy as np
import pandas as pd
from pycaret import regression, classification # For model setup and building
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix # For evaluating model performance
import time # For generating timestamps
from zipfile import ZipFile # For creating zip files


## Define the ModelBuilding class
class ModelBuilding:
    # Define class variables for numerical imputation methods and encoders
    NUM_IMP_METHODS = {"mean": "mean", "median":"median", "mode":"mode"}
    ENCODERS = {"OneHotEncoder": OneHotEncoder, 
                "LabelEncoder":OrdinalEncoder}

    ## Constructor to initialize the ModelBuilding object
    def __init__(self, dataset, target_column, classification_threshold=10):
        ## Initialize instance variables
        self.dataset = dataset
        self.target_column = target_column
        ## Initialize instance variables
        self.problem_type = self.get_problem_type\
        (dataset, target_column, classification_threshold=classification_threshold)

        ## Set the estimator based on the problem type
        self.estimator = self.estimator_setter()
        # Clean the dataset
        self.cleaned_dataset = self.clean_data(self.dataset, target_column)

    ## Method to set the estimator based on the problem type
    def estimator_setter(self):
        return regression if self.problem_type == "Regression" else classification
            
    ## Method to perform setup for model building
    def setup(self, 
              cat_imputation="mode", 
              num_imputation="mean", 
              encoder="OneHotEncoder", 
              train_size=.75, 
              polynomial_features=False,
              date_features=None,
              datetime_format="mixed",
              normalize=False,
              normalize_method="zscore"
             ):
        ## Handle numerical imputation methods
        num_imputation = self.NUM_IMP_METHODS.get(num_imputation, "mean")
        ## Initialize the encoder based on the specified method
        encoding_method = self.ENCODERS.get(encoder, OneHotEncoder)(handle_missing="return_nan",
                                                                    drop_invariant=True,
                                                                    handle_unknown="return_nan")

        ## Copy the dataset for processing
        self.tmp_dataset = self.cleaned_dataset.copy()
        ## Handle date features if provided
        if date_features:
            for col in date_features:
                tmp_col = self.tmp_dataset[col]
                try:
                    self.tmp_dataset[col] = pd.to_datetime(tmp_col, 
                                                                 format=datetime_format)
                except:
                    try:
                        tmp_col = tmp_col.map(lambda x: x.split(" ")[0])
                        self.tmp_dataset[col] = pd.to_datetime(tmp_col, 
                                                                     format=datetime_format.split(" ")[0])
                    except:
                        date_features.remove(col)
                        continue
        ## Perform setup using the Pycaret library
        self.experiment = self.estimator.setup(self.tmp_dataset, target=self.target_column, 
                                               remove_outliers=False, 
                                               numeric_imputation=num_imputation,
                                               categorical_imputation=cat_imputation,
                                               max_encoding_ohe=0,
                                               encoding_method=encoding_method,
                                               html=False,
                                               verbose=False,
                                               train_size=train_size,
                                               polynomial_features=polynomial_features,
                                               date_features=date_features,
                                               normalize=normalize,
                                               normalize_method=normalize_method)
        ## Format setup results
        self.setup_table = self.estimator.pull()\
        .style\
        .map(lambda val: 'background-color: green' if str(val)=="True" else None, subset=["Value"])
    
    ## Method to build the model
    def build(self, selected_models=None, evaluate_model=False):
        ## Compare models and select the best one
        self.best_model = self.estimator.compare_models(include=selected_models, verbose=False, n_select=1)
        self.compare_models = self.estimator.pull()
        ## Evaluate model performance if specified
        if evaluate_model:
            ## This function only works in IPython enabled Notebook
            self.estimator.evaluate_model(self.best_model)
        ## Finalize the best model pipeline
        self.best_model_pipeline = self.estimator.finalize_model(self.best_model)
        ## Format model comparison results
        if self.problem_type == "Regression":
            self.compare_models = self.compare_models.style.\
            highlight_max(subset=["R2"])\
            .highlight_min(subset=["MAE", "MSE", "RMSE", "RMSLE", "MAPE"])
        else:
            self.compare_models = self.compare_models.style\
            .applymap(lambda x: "background-color: gray", subset=["TT (Sec)"]).format(precision=3)\
            .background_gradient(subset=["Accuracy", "AUC", "Recall", "Prec.", "F1", "Kappa", "MCC"], axis=0)\
            .highlight_max(subset=["Accuracy", "AUC", "Recall", "Prec.", "F1", "Kappa", "MCC"], color="green")
        
        ## Evaluate model performance for classification problems
        if self.problem_type != "Regression":
            ticks, self.labels = tuple(pd.DataFrame(self.experiment.y_transformed.drop_duplicates()).merge(self.experiment.y,left_index=True, right_index=True).T.values)
            self.ticks=ticks.astype(int)+.5
            self.y_pre = self.best_model.predict(self.experiment.test_transformed.iloc[:,:-1])
            
            self.precision_recall_fscore_support = np.array(precision_recall_fscore_support(self.experiment.y_test_transformed,self.y_pre)).transpose()
            self.confusion_matrix = confusion_matrix(self.experiment.y_test_transformed,self.y_pre)
            
            ## Extract and format feature importances if available
            if "feature_importances_" in dir(self.best_model):
                self.feature_names = self.experiment.dataset_transformed.iloc[:,:-1].columns
                self.feature_importances= self.best_model.feature_importances_
                self.features = pd.DataFrame(zip(self.feature_names, self.feature_importances))\
                .sort_values(by=1, ascending=False)\
                .iloc[:10,:].reset_index()
                
            
        
    ## Method to download the model
    def download(self):
        # # Generate a timestamp for naming the zip file
        timestamp = f"{int(time.time())}"
        model_name = "model.joblib"
        pipeline_name = "pipeline.joblib"
        ## Dump the best model and pipeline to joblib files
        dump(self.best_model, model_name)
        dump(self.best_model_pipeline, pipeline_name)
        ## Create a zip file containing the model files
        with ZipFile("my_model_temp.zip", 'w') as myzip:
            myzip.write(model_name)
            myzip.write(pipeline_name)
        ## Read the zip file as bytes
        with open("my_model_temp.zip", 'rb') as f:
            model_bytes = f.read()
        
        # Return the model name and bytes
        return f"my_model_{timestamp}.zip", model_bytes

    ## Class method to determine the problem type based on the target column
    @classmethod
    def get_problem_type(cls, df, target_column, classification_threshold=10):
        target_dtype = df[target_column].dtype
        num_unique_values = df[target_column].nunique()
        ## Determine the problem type based on the target column properties
        if (target_dtype == "float64")\
         or (target_dtype == "int64" and num_unique_values > classification_threshold):
            return "Regression"
        elif num_unique_values == 2:
            return "Classification (Binary)"
        elif num_unique_values > 2:
            return "Classification (Multiclass)" 
        else:
            raise ValueError("Target Columns must have at least 2 unique values")
    
    ## Class method to clean the dataset
    @classmethod
    def clean_data(cls, df, target_column):
        ## Remove duplicate rows
        df = df.drop_duplicates()
        ## Remove rows with missing values in the target column
        df = df.loc[df[target_column].notna()]
        ## Remove columns with only one unique value
        for col in df.columns:
            if len(df[col].unique()) == 1:
                df.drop(col, axis=1)
        
        return df
        
