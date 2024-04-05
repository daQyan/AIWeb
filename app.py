import streamlit as st
from transformers import pipeline
import pandas as pd

# Load the language model
model = pipeline("text-generation", model="openai-gpt")

# Define Streamlit app
def main():
    st.title("Language Model Deployment with Streamlit")

    # App Description
    st.markdown("""
        ## Explore AI Text Generation
        Enter text below and see how the AI model continues it.
    """)

    # Custom CSS to inject for styling
    st.markdown("""
        <style>
        .stTextArea > div > div > textarea {
            background-color: #f4f4f9; /* Light gray background */
            color: #444; /* Darker text color */
        }
        </style>
        """, unsafe_allow_html=True)

    # Text input with custom color
    text_input = st.text_area("Enter text to generate continuation:", height=150)

    # Generate text button
    if st.button("Generate Text"):
        generate_text(text_input)

    # Optional: Display a chart comparing text lengths
    st.sidebar.header("Text Length Analysis")
    st.sidebar.bar_chart(create_length_comparison_chart(text_input))

def generate_text(input_text):
    if input_text:
        with st.spinner('AI is at work...'):
            generated_text = model(input_text, max_length=50, do_sample=True)[0]['generated_text']
        
        st.subheader("Generated Text:")
        st.write(generated_text)
    else:
        st.warning("Please enter some text first.")

def create_length_comparison_chart(input_text):
    data = {'Original Length': len(input_text.split()), 'Generated Length': len(input_text.split()) * 2}  # Simplified example
    return pd.DataFrame(data, index=['Length'])

if __name__ == "__main__":
    main()
