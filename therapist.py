from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv
import os
import shelve

# Simple Web Version Of The Therapist AI ChatBot
# TO DO
# The alignment of the messages will be arranged.
# The previous conservations will be stored in a database.
# Color combinations will be updated.


load_dotenv()
st.title("MelodiCell")

st.markdown(
    """
    <style>
      /* Upper Rectangle (Header) → Green */
      header[data-testid="stHeader"] {
        background-color: #FFCC00 !important;
      }

      /* Original styles below */
      section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
      }

      div[data-testid="stAppViewContainer"] {
        background-color: #0051A2 !important;
      }

      div[data-testid="main"] {
        background-color: transparent !important;
      }
      }
    </style>
    """,
    unsafe_allow_html=True,
)

USER_AVATAR = "👤"
BOT_AVATAR = "🧑‍⚕️"
client = OpenAI(api_key="")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"


def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages", [])


def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages


if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

with st.sidebar:
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        save_chat_history([])

for message in st.session_state.messages:
    avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""
        for response in client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=st.session_state["messages"],
            stream=True,
        ):
            full_response += response.choices[0].delta.content or ""
            message_placeholder.markdown(full_response + "|")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Save chat history after each interaction
save_chat_history(st.session_state.messages)