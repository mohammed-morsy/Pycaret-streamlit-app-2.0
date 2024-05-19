## Import necessary libraries
import streamlit as st
import matplotlib.pyplot as plt

## Define a function to plot a figure with customizable grid and rotation options
def plot(fig, grid=False):
    ## Check if the current figure has any axes; if not, rerun the Streamlit app
    if not plt.gcf().axes:
        st.rerun()
    
    ## Create an empty placeholder for dynamic content
    placeholder = st.empty()
    
    ## Create two columns for user input sliders
    col1, col2 = st.columns(2)
    
    ## Slider for adjusting X-axis tick rotation
    with col1:
        x_rotation = st.slider(":green[X ticks rotation]", min_value=0, max_value=360, value=0)
    ## Slider for adjusting X-axis tick rotation
    with col2:
        y_rotation = st.slider(":green[Y ticks rotation]", min_value=0, max_value=360, value=0)
    
    ## Create another two columns for user input sliders
    col3, col4 = st.columns(2)
    ## Slider for adjusting X-axis label rotation
    with col3:
        x_label_rotation = st.slider(":green[X labels rotation]", min_value=0, max_value=360, value=0)
    ## Slider for adjusting Y-axis label rotation
    with col4:
        y_label_rotation = st.slider(":green[Y labels rotation]", min_value=0, max_value=360, value=90)
    
    ## Apply the user-specified rotations and grid settings to each axis in the figure
    for ax in fig.axes:
        ax.grid(visible=grid) # Set grid visibility
        ax.xaxis.label.set_rotation(x_label_rotation) # Rotate X-axis labels
        ax.yaxis.label.set_rotation(y_label_rotation) # Rotate Y-axis labels
        ax.yaxis.label.set_ha('right') # Align Y-axis labels to the right
        ax.tick_params(axis="x", rotation=x_rotation) # Rotate X-axis ticks
        ax.tick_params(axis="y", rotation=y_rotation) # Rotate Y-axis ticks
    ## Display the plot within the placeholder container
    with placeholder.container():
        st.pyplot(fig, transparent=True, edgecolor="black")

## Define a simpler plotting function without grid and rotation customization
def plot2(fig):
    # Check if the current figure has any axes; if not, rerun the Streamlit app
    if not plt.gcf().axes:
        st.rerun()
    # Set the grid visibility to False
    plt.grid(False)
    # Display the plot
    st.pyplot(fig, transparent=True, edgecolor="black")
