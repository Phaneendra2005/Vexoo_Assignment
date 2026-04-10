import streamlit as st

st.title("Vexoo AI Assignment")

query = st.text_input("Enter your query")

if query:
    # Replace this with your retrieval logic
    result = f"Processed query: {query}"
    st.write(result)