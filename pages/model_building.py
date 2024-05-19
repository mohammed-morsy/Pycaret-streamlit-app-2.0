from model_building import ModelBuilding
import streamlit as st
from datavisualizer import ModelEvaluation
from st_pages import add_page_title
import os
import sys
parent = os.getcwd()
sys.path.append(parent)

from plot import plot2

add_page_title()


def main():
    # st.markdown(homepage, unsafe_allow_html=True)
    st.markdown("<h1 style='color:gray; text-align:center;'>Supervised Learning</h1>", unsafe_allow_html=True)
    
    st.sidebar.markdown("<h1 style='tex-align:center; color:#05211b;'>Choose Columns</h1>", unsafe_allow_html=True)
    selected_columns = st.sidebar.multiselect(":green[Available Columns]", 
                                              st.session_state.data.columns,
                                              default=st.session_state.data.columns.tolist())
    
    target_column = st.sidebar.selectbox(":green[Select the target column:]", 
                                         options=selected_columns,
                                         index=len(selected_columns)-1)
    tmp_data = st.session_state.data[selected_columns].copy()
    
    st.sidebar.button("Commit Changes", on_click=commit_changes_button, use_container_width=True)
    
    # Check if any of the following conditions are true:
    # 1. The 'commit_changes_clicked' flag is not present in the session state (initial state).
    # 2. The 'commit_changes_clicked' flag is set to True (indicating the user clicked the commit button).
    # 3. The selected columns have changed (i.e., the currently selected columns are different from the model's dataset columns).
    #    If 'selected_columns' is not in the session state, default to True to ensure the model rebuilds initially.
    if ("commit_changes_clicked" not in st.session_state) \
        or (st.session_state.commit_changes_clicked) \
        or ((sorted(st.session_state.selected_columns) != sorted(st.session_state.model.dataset.columns)) \
            if "selected_columns" in st.session_state else True):
        # Update the session state with the currently selected columns.
        st.session_state.selected_columns = selected_columns
        # Rebuild the model with the updated dataset and target column.
        st.session_state.model = ModelBuilding(tmp_data, target_column)
        # Reset the 'commit_changes_clicked' flag.
        st.session_state.commit_changes_clicked = False

        

    if "model" in st.session_state:
        st.markdown(f"""<h3 style='color: #030214;'>Problem Type</h3> 
        <p style='color:#05211b;'>{st.session_state.model.problem_type}</p>""", 
                    unsafe_allow_html=True)
        
        model_setup()
        
        if "experiment" in dir(st.session_state.model):            
            model_building()
            if "best_model" in dir(st.session_state.model):
                st.success(":green[Model Building Completed]", icon="✅")
                # Display model information
                download_model()

                st.markdown("<h2 style='text-align:center; color: #030214;'>Model Comparison results</h2>", 
                            unsafe_allow_html=True)
                st.dataframe(st.session_state.model.compare_models, use_container_width=True)
                
                model = ModelEvaluation()
                fig = model.bar(st.session_state.model.compare_models.data)
                plot2(fig)
                
                st.markdown("<h2 style='color: #030214;'>Best Model pipeline</h2>", unsafe_allow_html=True)
                st.write(st.session_state.model.best_model_pipeline)
                model_metrics()
                
                
                                        
    
def commit_changes_button():
    st.session_state.commit_changes_clicked = True
    
def setup_button():
    st.session_state.setup_clicked = True
    st.session_state.build_clicked = False
def build_button():
    st.session_state.build_clicked = True

def model_setup():
    cat_imp_methods = ["mode", "add_class"]
    st.markdown("<h2 style='color: #030214;'>Preprocessing parameters</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # st.subheader("Filling methods")
        cat_imputation = st.selectbox("For categorical columns:", 
                                     options=cat_imp_methods,
                                     index=0)
        if cat_imputation == "add_class":
            cat_imputation = st.text_input("custom class", value="missing_value")
        
        num_imp_methods = ("mean", "median", "mode")
        num_imputation = st.selectbox("For numerical columns:", 
                                     options=num_imp_methods,
                                     index=0)
    with col2:
        # st.subheader("Handle categorical columns")
        encoders = ("LabelEncoder", "OneHotEncoder")
        encoder = st.selectbox("Encoder type:", 
                                       options=encoders,
                                       index=0)
        
        train_size = st.number_input("training set proportion", min_value=.0, max_value=1., value=.75)
    with col3:
        feature_cols = st.session_state.data.columns.tolist()
        feature_cols.remove(st.session_state.model.target_column)
        date_features = st.multiselect("Date features", feature_cols,
                                               default=None)

        datetime_format = (lambda x: st.\
                           text_input("format eg. \"%Y/%d/%m\"", value="mixed") if x else None)(date_features)
    with col4:
        normalize = st.selectbox("Normalize", options=["Yes", "No"], index=1) == "Yes"
        normalize_method = (lambda x: st.selectbox("Normalization method", 
                                            options=["zscore", "minmax", "maxabs", "robust"], 
                                            index=0) if normalize else "zscore")(normalize)      

    if "experiment" not in  dir(st.session_state.model):
        st.dataframe(st.session_state.model.dataset, use_container_width=True)
    
    if "setup_clicked" not in st.session_state:
        st.session_state.setup_clicked = False
    
    st.button("Setup", on_click=setup_button, use_container_width=True)
    if st.session_state.setup_clicked:
        with st.spinner("Performing Model Setup..."):
            st.session_state.model.setup(cat_imputation=cat_imputation, 
                                         num_imputation=num_imputation, 
                                         encoder=encoder, 
                                         train_size=train_size,
                                         date_features=date_features,
                                         datetime_format=datetime_format,
                                         normalize=normalize,
                                         normalize_method=normalize_method)
        st.session_state.setup_clicked=False
            
    if "experiment" in dir(st.session_state.model):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h3 style='text-align:center; color:#05211b;'>Transformed Data</h3>", unsafe_allow_html=True)
            st.dataframe(st.session_state.model\
                         .experiment.dataset_transformed.sort_index()\
                         .style.map(lambda x: "background-color:gray", 
                                    subset=[st.session_state.model.target_column]), 
                         use_container_width=True)
        with col2:
            st.markdown("<h3 style='text-align:center; color:#05211b;'>Cleaned Data</h3>", unsafe_allow_html=True)
            st.dataframe(st.session_state.model\
                         .tmp_dataset.sort_index()\
                         .style.map(lambda x: "background-color:gray", 
                                    subset=[st.session_state.model.target_column]), 
                         use_container_width=True)
        st.dataframe(st.session_state.model.setup_table, use_container_width=True)

    


def model_building():    
    # Define available models based on problem type
    if st.session_state.model.problem_type == "Regression":
        available_models = ["lr", "lasso", "ridge", "rf", "knn", "dt", "svm"]
    else:
        available_models = ["lr", "dt", "rf", "svm", "knn"]

    # Allow user to select models
    selected_models = st.multiselect("Select models (2 to 5):", available_models, default=available_models[:5], key="selected_models")
    
    # Check if the number of selected models is within the allowed range
    if len(selected_models) < 2 or len(selected_models) > 5:
        st.warning("Please select between 2 and 5 models.")
        return
    

    # Show button to start model building
    if "build_clicked" not in st.session_state:
        st.session_state.build_clicked = False
        
    st.button("Start building", on_click=build_button, use_container_width=True)
    if st.session_state.build_clicked:
        with st.spinner("Performing Model Building..."):
            st.session_state.model.build(selected_models)
        st.session_state.build_clicked = False
    
        


def download_model():
    model_name, model_bytes = st.session_state.model.download()

    download = st.download_button(label="Download Model", 
                                  data=model_bytes, 
                                  file_name=model_name, 
                                  mime="application/octet-stream", 
                                  use_container_width=True)
    
    # Instructions to load the model
    st.markdown("""
    To load the downloaded model, please ensure that you have `joblib` version 1.3.2 installed. You can install it using pip:

    ```shell
    pip install joblib==1.3.2
    ```
    ```python
    import joblib
    ```
    Then, you can load the model in your Python script like this:
    ```python
    # Replace 'my_model.joblib' with the path to your downloaded model file
    model = joblib.load('my_model.joblib')
    ```
    """)


def model_metrics():
    # Display model information
    if st.session_state.model.problem_type != "Regression":
        st.markdown("<h1 style='color:gray; text-align:center;'>Model Evaluation</h1>", unsafe_allow_html=True)
        model = ModelEvaluation()
        
        st.markdown("""<h2 style='color: #030214;'>Confusion Matrix:</h2>
        <br><p>- rows represent the actual classes and columns represent the predicted classes</p>
        """, 
                    unsafe_allow_html=True)
        
        fig = model.confusion_matrix(st.session_state.model.confusion_matrix, 
                                     st.session_state.model.ticks, 
                                     st.session_state.model.labels)
        plot2(fig)
    
        st.markdown(f"""<h2 style='color: #030214;'>Classification report:</h2>
        <br><p>This report shows the following metrics related to each class</p>
        """, 
                    unsafe_allow_html=True)
    
        st.markdown("<h5 style='color:#05211b;'>Precision: </h5>",
                    unsafe_allow_html=True)
        st.latex(r"\frac{True_{positive}}{True_{positive}+False_{positive}}")
        
        st.markdown("<h5 style='color:#05211b;'>Rcall: </h5>",
                    unsafe_allow_html=True)
        st.latex(r"\frac{True_{positive}}{True_{positive}+False_{negative}}")
        
        st.markdown("<h5 style='color:#05211b;'>F1 score(Harmonic mean of precision and recall): </h5>",
                    unsafe_allow_html=True)
        st.latex(r"\frac{2*Precision*Recall}{Precision+Recall}")
    
        st.markdown("<h5 style='color:#05211b;'>Support: The number of occurrences of each class in y_test</h5>",
                    unsafe_allow_html=True)
        
        
        fig = model.classification_report(st.session_state.model.precision_recall_fscore_support, 
                                          st.session_state.model.ticks, 
                                          st.session_state.model.labels)
        plot2(fig)
        
        if "feature_importances" in dir(st.session_state.model):
            fig = model.hlines(st.session_state.model.features, 
                               st.session_state.model.feature_names)
            plot2(fig)
        


if "data" in st.session_state and st.session_state.data is not None:
    main()
else:
    st.switch_page("main.py")
