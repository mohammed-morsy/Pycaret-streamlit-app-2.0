## Import necessary modules and functions from external files and packages
from dataloader import DataLoader
import streamlit as st
from st_pages import Page, Section, add_page_title, show_pages

## Add the child directory `pages` to the system path 
## to enable importing modules from the it
import os, sys
sys.path.append(os.getcwd()+"\\pages")
from datavisualizer import ModelEvaluation

## Define and show the pages available in the Streamlit app
show_pages(
    [
        Page("main.py", "Home", "🏠"),
        Page("pages/eda.py", "Exploratory Data Analysis", "📈"),
        Page("pages/model_building.py", "Model Building", "⚙️"),
    ]
)

## Add a title `🏠 Home` to the Streamlit page
add_page_title()


st.markdown("<h1 style='text-align:center; color:gray;'>Auto Machine Learning &<br>Data Exploration</h1>", unsafe_allow_html=True)
def main():
    
    ## Initialize the DataLoader if it is not already in the session state    
    if "data_loader" not in st.session_state:
        st.session_state.data_loader = DataLoader()
    
    ## Load data
    st.session_state.data = st.session_state.data_loader.load_data()
    
    ## If data is loaded successfully, display a sample of the dataset
    if st.session_state.data is not None:
        st.markdown("<h2 style='text-align:center;color: #8080c0;'>Loaded Dataset Sample</h2>", unsafe_allow_html=True)
        st.dataframe(st.session_state.data.sample(min(10,st.session_state.data.shape[0])), use_container_width=True)
        
        ## Add a sidebar with an action menu for navigating between pages
        st.sidebar.markdown("<h1 style='text-align:center; color:#05211b;'>Action Menu</h1>", unsafe_allow_html=True)
        
        ## Create two columns in the sidebar for different actions
        col1, col2 = st.sidebar.columns(2)
        ## Add a button for navigating to the EDA page in the first column
        with col1:
            st.markdown("<h2 style='text-align:center; color:#05211b;'>Exploratory Data Analysis</h2>", unsafe_allow_html=True)
            if st.button("Exploratory Data Analysis"):
                st.switch_page("pages/eda.py")
        ## Add a button for navigating to the Model Building page in the second column
        with col2:
            st.markdown("<h2 style='text-align:center; color:#05211b;'>Supervised Learning</h2>", unsafe_allow_html=True)
            if st.button("Supervised Learning"):
                st.switch_page("pages/model_building.py")
        

## Run the main function when the script is executed
if __name__ == "__main__":
    main()
