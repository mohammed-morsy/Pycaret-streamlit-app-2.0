import streamlit as st
import matplotlib.pyplot as plt

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

def plot2(fig):
    if not plt.gcf().axes:
        st.rerun()
    plt.grid(False)
    st.pyplot(fig, transparent=True, edgecolor="black")