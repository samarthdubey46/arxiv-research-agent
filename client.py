import streamlit as st
st.text_input("Question", key="Enter Your question")
print(st.session_state.name)

