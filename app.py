import streamlit as st
from dotenv import load_dotenv

load_dotenv("secret.env")

st.title("Mini AI Knowledge Assistant")
st.write("Upload a PDF and ask questions about it.")
