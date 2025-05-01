import streamlit as st
from transformers import pipeline

st.title("🧠 TinyLlama Chatbot")

# Load TinyLlama model only once
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

generator = load_model()

# System prompt to set behavior
system_message = "<|system|>\nYou are a helpful assistant.\n"

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask me anything..."):
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Build full TinyLlama-style prompt
    full_prompt = system_message
    for msg in st.session_state.messages:
        tag = "<|user|>" if msg["role"] == "user" else "<|assistant|>"
        full_prompt += f"{tag}\n{msg['content']}\n"
    full_prompt += "<|assistant|>\n"

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        result = generator(full_prompt, max_length=512, num_return_sequences=1, truncation=True)[0]['generated_text']
        # Extract assistant reply
        reply = result.split("<|assistant|>")[-1].strip()
        message_placeholder.markdown(reply)

    # Save assistant reply
    st.session_state.messages.append({"role": "assistant", "content": reply})
