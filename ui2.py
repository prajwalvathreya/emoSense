import streamlit as st

# Initialize session state if not already initialized
if 'count' not in st.session_state:
    st.session_state.count = 0

# Function to update count value
def increment_counter():
    st.session_state.count += 1

# UI Elements
st.title("Session State Demo")

# Display current counter value
st.write(f"Current counter value: {st.session_state.count}")

# Button to increment counter
if st.button("Increment"):
    increment_counter()
    st.rerun()  # Rerun the script to reflect the changes immediately

