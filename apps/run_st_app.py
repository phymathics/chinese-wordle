import os
import streamlit as st

st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊",
    # layout="wide",
    initial_sidebar_state="expanded",
)

os.system("streamlit run st_app.py")
