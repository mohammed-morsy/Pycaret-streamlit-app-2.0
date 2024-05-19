import os
import sys
parent = os.getcwd()
sys.path.append(parent)
from plot import plot
from datavisualizer import DataVisualizer
import streamlit as st
import matplotlib.pyplot as plt
from st_pages import add_page_title

add_page_title()


def main():    
    st.markdown("<h1 style='color: gray; text-align:center;'>Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    st.dataframe(st.session_state.data, use_container_width=True)
    actions = ["Data Visualizations", "Data Summary"]
    st.markdown("<h3 style='text-align: center; color: #030214;'>Analysis Type</h3>", unsafe_allow_html=True)
    eda_option = st.multiselect(":green[select Type:]", actions, default=actions)
    
    if "Data Summary" in eda_option:
        st.markdown("<h2 style='text-align:center; color: #030214;'>Data Summary</h2>", unsafe_allow_html=True)
        
        if st.session_state.data.select_dtypes(include=["number"]).empty:
            st.warning("No numerical columns found.")
        else:
            describe = st.session_state.data.describe()
            st.markdown("<h3 style='color: #05211b;'>Numerical Columns</h3>", unsafe_allow_html=True)
            st.dataframe(describe, use_container_width=True)
            
            if st.session_state.data.select_dtypes(exclude="object").shape[1] > 1:
                st.markdown("<h3 style='color:#05211b;'>Covariance Matrix</h3>", unsafe_allow_html=True)
                st.pyplot(DataVisualizer.covariance_matrix(st.session_state.data.select_dtypes(exclude="object")), 
                          transparent=True, 
                          edgecolor="black")
    
        if st.session_state.data.select_dtypes(include=["object"]).empty:
            st.warning("No categorical columns found.")
        else:
            describe_cat = st.session_state.data.describe(include="object")
            st.markdown("<h3 style='color: #05211b;'>Categorical Columns</h3>", unsafe_allow_html=True)
            st.dataframe(describe_cat, use_container_width=True)
    
           
    # Visualizations
    if "Data Visualizations" in eda_option:
        num_cols = st.session_state.data.select_dtypes(exclude="object").columns.tolist()
        st.markdown("<h2 style='text-align:center; color: #030214;'>Custom Visualizations</h2>", unsafe_allow_html=True)
        
        visualizer = DataVisualizer(st.session_state.data)
        st.sidebar.markdown("<h1 style='color:#05211b'>Plot Type</h1>", unsafe_allow_html=True)
        plot_type = st.sidebar.radio(":green[Select plot type:]", ["Scatter Plot", "Histogram", "Box Plot", "Bar Plot", "Heatmap", "Line Plot", "Pairplot"])
        
        if plot_type == "Scatter Plot" or plot_type == "Box Plot" or plot_type == "Bar Plot":
            col1, col2, col3 = st.columns(3)
            with col1:
                x_variable = st.selectbox(":green[Select x variable:]", st.session_state.data.columns)
            with col2:
                y_variable = st.selectbox(":green[Select y variable:]", st.session_state.data.columns)
            with col3:
                grid = st.radio(":green[Show grid lines:]", ["Yes", "No"], index=1, horizontal=True) == "Yes"
            fig = getattr(visualizer, plot_type.lower().replace(" ", "_"))(x_variable, y_variable, grid=bool(grid))
            plot(fig, grid)
        elif plot_type == "Line Plot":
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                x_variable = st.selectbox(":green[Select x variable:]", st.session_state.data.columns)
            with col2:
                y_variable = st.selectbox(":green[Select y variable:]", st.session_state.data.columns, )
            with col3:
                scale = st.selectbox(":green[Select scale:]", [None, "linear", "log", "symlog", "logit"])
            with col4:
                grid = st.radio(":green[Show grid lines:]", ["Yes", "No"], index=1, horizontal=True) == "Yes"
            fig = visualizer.line_plot(x_variable, y_variable, scale=scale, grid=bool(grid))
            plot(fig, grid)
        elif plot_type == "Histogram":
            col1, col2, col3 = st.columns(3)
            with col1:
                selected_variable = st.selectbox(":green[Select variable for histogram:]", st.session_state.data.columns)
            with col2:
                bins = st.slider(":green[Number of bins:]", min_value=1, max_value=100, value=10)
            with col3:
                grid = st.radio(":green[Show grid lines:]", ["Yes", "No"], index=1, horizontal=True) == "Yes"
            fig = visualizer.histogram(selected_variable, bins, grid=bool(grid))
            plot(fig, grid)
        elif plot_type == "Heatmap":
            st.write("Select variables for the heatmap:")
            heatmap_variables = st.multiselect(":green[Select variables:]", num_cols, default=num_cols[:2])
            if heatmap_variables:
                fig = visualizer.heatmap(heatmap_variables)    
                plot(fig)
            else:
                st.warning("Please select at least one variable for the heatmap.")
        elif plot_type == "Pairplot":
            pairplot_variables = st.multiselect(":green[Select variables:]", num_cols, default=num_cols)
            col1, col2 = st.columns(2)
            with col1:
                diagonal = st.radio(":green[diagonal plot:]", ["hist", "kde"], index=0, horizontal=True)
            with col2:
                grid = st.radio(":green[Show grid lines:]", ["Yes", "No"], index=1, horizontal=True) == "Yes"
            fig = visualizer.pairplot(pairplot_variables, diagonal)
            plot(fig, grid)

def plot(fig, grid=False):
    if not plt.gcf().axes:
        st.rerun()
    placeholder = st.empty()
    plt.grid(grid)
    col1, col2 = st.columns(2)
    with col1:
        x_rotation = st.slider(":green[X ticks rotation]", min_value=0, max_value=360, value=0)
    with col2:
        y_rotation = st.slider(":green[Y ticks rotation]", min_value=0, max_value=360, value=0)
    col3, col4 = st.columns(2)
    with col3:
        x_label_rotation = st.slider(":green[X labels rotation]", min_value=0, max_value=360, value=0)
    with col4:
        y_label_rotation = st.slider(":green[Y labels rotation]", min_value=0, max_value=360, value=90)
    for ax in fig.axes:
        ax.grid(visible=grid)
        ax.xaxis.label.set_rotation(x_label_rotation)
        ax.yaxis.label.set_rotation(y_label_rotation)
        ax.yaxis.label.set_ha('right')
        ax.tick_params(axis="x", rotation=x_rotation)
        ax.tick_params(axis="y", rotation=y_rotation)
    with placeholder.container():
        st.pyplot(fig, transparent=True, edgecolor="black")
    
if "data" in st.session_state and st.session_state.data is not None:
    main()
else:
    st.switch_page("main.py")
