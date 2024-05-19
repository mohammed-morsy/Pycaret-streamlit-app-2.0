import pandas as pd
import streamlit as st
from pycaret.datasets import get_data

class DataLoader:
    """
    A class to facilitate data loading from various sources, including user-provided files
    and PyCaret's built-in datasets.
    """

    # Dictionary mapping file extensions to their corresponding formats
    FILE_EXTENSIONS = {
        "csv": "csv",
        "json": "json",
        "xls": "excel",
        "xlsx": "excel",
        "xlsm": "excel",
        "xlsb": "excel",
        "odf": "excel",
        "ods": "excel",
        "odt": "excel",
        "htm": "html",
        "html": "html",
        "sql": "sql",
        "parquet": "parquet",
        "feather": "feather",
        "dta": "stata",
        "sas7bdat": "sas",
        "sav": "spss",
        "orc": "orc",
        "gbq": "google bigquery",
        "h5": "hdf5",
        "hdf5": "hdf5",
        "hdf": "hdf5",
        "msg": "msgpack",
        "msgpack": "msgpack",
        "pkl": "pickle",
        "pickle": "pickle",
        "avro": "avro",
        "xml": "xml"
    }   

    # List of problem types for dataset selection
    PROBLEM_TYPES = ["Classification (Binary)", "Classification (Multiclass)", "Regression"]

    def __init__(self):
        """
        Initializes the DataLoader class.

        Fetches PyCaret's datasets and filters for classification and regression datasets.
        """
        pycaret_data =  get_data("index", verbose=False)
        self.cla_reg = pycaret_data.loc[(pycaret_data["Default Task"].isin(self.PROBLEM_TYPES)) & (pycaret_data["# Instances"] < 10000)]
        self.pycaret_datasets = self.cla_reg["Dataset"].values

    def load_data(self, user_file=None):
        """
        Load data from user-provided file or PyCaret's built-in datasets.

        Parameters:
        - user_file (DataFrame or None): DataFrame uploaded by the user.
        
        Returns:
        - DataFrame: Loaded dataset.
        """
        # Allow users to upload a file
        self.user_file = st.file_uploader("Upload your data file (CSV, Excel, etc.):")

        # Provide a selection box for PyCaret datasets
        pycaret_dataset = st.selectbox("Select a PyCaret dataset:", self.pycaret_datasets)
        with st.spinner("Loading..."):
            if self.user_file is not None:
                try:
                    # Load user-provided data
                    data = self.read_file()  
                    st.success("User's data loaded successfully", icon="✅")
                    return data
                except Exception as e:
                    st.error(f"Error loading user's data: {e}")
            elif pycaret_dataset:
                try:
                    # Load PyCaret's built-in dataset
                    data = get_data(pycaret_dataset, verbose=False)
                    st.success(f"PyCaret dataset '{pycaret_dataset}' loaded successfully", icon="✅")
                    return data
                except Exception as e:
                    st.error("Error loading PyCaret dataset: {e}")
            else:
                st.warning("Please provide either user_file or pycaret_dataset.")

    def read_file(self):
        """
        Read a file based on its extension.

        Returns:
        - DataFrame: Loaded dataset.
        """
        if self.user_file is not None:
            try:
                file_extension = self.user_file.name.split(".")[-1].lower()
                file_format = self.FILE_EXTENSIONS.get(file_extension)
                if not file_format:
                    st.error("Unsupported file format.")
                    st.write("Supported file formate:\n    -", '\n    -'.join(sorted(set(cls.FILE_EXTENSIONS.values()))))
                else:
                    read_function = getattr(pd, f"read_{file_format}")
                    df = read_function(self.user_file)
                    return df
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

