import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st
from ai_agent.agent import interact_with_agent
from langchain.memory import ConversationBufferMemory

st.set_page_config(
    page_title="Predictive Maintenance Agent 🤖",
    layout="wide",
)

def run_app():
    st.title("Predictive Maintenance AI Agent 🤖")
    
    col1, _ = st.columns([0.5, 1])
    col1.text_input(label='Enter your OpenAI API Key', key='openAI_api_key', type='password')
    if st.session_state.get('openAI_api_key', None):
    
        if st.button("New Chat"):
            st.session_state.pop('messages', None)
            st.session_state.pop('memory', None)
            st.session_state.pop('openAI_api_key', None)
            st.rerun()
        
        agent_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        st.session_state.setdefault('memory', agent_memory)
        st.session_state.setdefault('messages', [])

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if user_query := st.chat_input("Ask a question:"):
            with st.chat_message("user"):
                st.markdown(user_query)
            st.session_state.messages.append({"role": "user", "content": user_query})

            with st.spinner("Thinking..."):
                try:
                    response = interact_with_agent(
                        query=user_query,
                        memory=st.session_state['memory'],
                        openAI_api_key=st.session_state['openAI_api_key']
                        )
                    with st.chat_message("assistant"):
                        st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
                except Exception as e:
                    st.write(str(e))

            
    return 


if __name__ == '__main__':
    run_app()