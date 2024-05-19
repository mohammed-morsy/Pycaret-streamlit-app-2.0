from preprocessing import clean_data, get_problem_type
from pycaret import regression, classification
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from joblib import dump
from category_encoders.one_hot import OneHotEncoder
from category_encoders.ordinal import OrdinalEncoder
import time
import numpy as np
from zipfile import ZipFile
import pandas as pd



class ModelBuilding:
    NUM_IMP_METHODS = {"mean": "mean", "median":"median", "mode":"mode"}
    ENCODERS = {"OneHotEncoder": OneHotEncoder, 
                "LabelEncoder":OrdinalEncoder}

    
    def __init__(self, dataset, target_column):
        self.dataset = dataset
        self.target_column = target_column
        self.problem_type = get_problem_type(dataset, target_column)
        self.estimator = self.estimator_setter()
        self.processed_dataset = clean_data(self.dataset, target_column)

    
    def estimator_setter(self):
        return regression if self.problem_type == "Regression" else classification
            
    
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
        num_imputation = self.NUM_IMP_METHODS.get(num_imputation, "mean")
        encoding_method = self.ENCODERS.get(encoder, OneHotEncoder)(handle_missing="return_nan",
                                                                    drop_invariant=True,
                                                                    handle_unknown="return_nan")
        self.tmp_dataset = self.processed_dataset.copy()
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
        self.setup_table = self.estimator.pull()\
        .style\
        .map(lambda val: 'background-color: green' if str(val)=="True" else None, subset=["Value"])

    def build(self, selected_models=None, evaluate_model=False):
        # Compare Models
        self.best_model = self.estimator.compare_models(include=selected_models, verbose=False, n_select=1)
        self.compare_models = self.estimator.pull()
        # Evaluate Model
        
        if evaluate_model:
            ## This function only works in IPython enabled Notebook
            self.estimator.evaluate_model(self.best_model)
        # Deploy Model
        self.best_model_pipeline = self.estimator.finalize_model(self.best_model)
        if self.problem_type == "Regression":
            self.compare_models = self.compare_models.style.\
            highlight_max(subset=["R2"])\
            .highlight_min(subset=["MAE", "MSE", "RMSE", "RMSLE", "MAPE"])
        else:
            self.compare_models = self.compare_models.style\
            .applymap(lambda x: "background-color: gray", subset=["TT (Sec)"]).format(precision=3)\
            .background_gradient(subset=["Accuracy", "AUC", "Recall", "Prec.", "F1", "Kappa", "MCC"], axis=0)\
            .highlight_max(subset=["Accuracy", "AUC", "Recall", "Prec.", "F1", "Kappa", "MCC"], color="green")
        
        if self.problem_type != "Regression":
            ticks, self.labels = tuple(pd.DataFrame(self.experiment.y_transformed.drop_duplicates()).merge(self.experiment.y,left_index=True, right_index=True).T.values)
            self.ticks=ticks.astype(int)+.5
            self.y_pre = self.best_model.predict(self.experiment.test_transformed.iloc[:,:-1])
            
            self.precision_recall_fscore_support = np.array(precision_recall_fscore_support(self.experiment.y_test_transformed,self.y_pre)).transpose()
            self.confusion_matrix = confusion_matrix(self.experiment.y_test_transformed,self.y_pre)
            
            
            if "feature_importances_" in dir(self.best_model):
                self.feature_names = self.experiment.dataset_transformed.iloc[:,:-1].columns
                self.feature_importances= self.best_model.feature_importances_
                self.features = pd.DataFrame(zip(self.feature_names, self.feature_importances))\
                .sort_values(by=1, ascending=False)\
                .iloc[:10,:].reset_index()
                
            
        

    def download(self):
        timestamp = f"{int(time.time())}"
        model_name = "model.joblib"
        pipeline_name = "pipeline.joblib"
        dump(self.best_model, model_name)
        dump(self.best_model_pipeline, pipeline_name)
        
        with ZipFile("my_model_temp.zip", 'w') as myzip:
            myzip.write(model_name)
            myzip.write(pipeline_name)

        with open("my_model_temp.zip", 'rb') as f:
            model_bytes = f.read()

        return f"my_model_{timestamp}.zip", model_bytes

