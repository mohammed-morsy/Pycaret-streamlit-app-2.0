# from css import homepage
import streamlit as st
from dataloader import DataLoader
from st_pages import Page, Section, add_page_title, show_pages

import os, sys
sys.path.append(os.getcwd()+"\\pages")
from datavisualizer import ModelEvaluation

add_page_title()
show_pages(
    [
        Page("main.py", "Home", "🏠"),
        Page("pages/eda.py", "Exploratory Data Analysis", "📈"),
        Page("pages/model_building.py", "Model Building", "⚙️"),
    ]
)

st.markdown("<h1 style='text-align:center; color:gray;'>Auto Machine Learning &<br>Data Exploration</h1>", unsafe_allow_html=True)
def main():
    palceholder = st.empty()
    # Initialize DataLoader
    if "data_loader" not in st.session_state:
        st.session_state.data_loader = DataLoader()
    
    # Load data
    st.session_state.data = st.session_state.data_loader.load_data()
    
    if st.session_state.data is not None:
        st.markdown("<h2 style='text-align:center;color: #8080c0;'>Loaded Dataset Sample</h2>", unsafe_allow_html=True)
        st.dataframe(st.session_state.data.sample(min(10,st.session_state.data.shape[0])), use_container_width=True)
        
        st.sidebar.markdown("<h1 style='text-align:center; color:#05211b;'>Action Menu</h1>", unsafe_allow_html=True)
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.markdown("<h2 style='text-align:center; color:#05211b;'>Exploratory Data Analysis</h2>", unsafe_allow_html=True)
            if st.button("Exploratory Data Analysis"):
                st.switch_page("pages/eda.py")
        with col2:
            st.markdown("<h2 style='text-align:center; color:#05211b;'>Supervised Learning</h2>", unsafe_allow_html=True)
            if st.button("Supervised Learning"):
                st.switch_page("pages/model_building.py")
        


if __name__ == "__main__":
    main()
