import streamlit as st
import os

from utils.document_processor import DocumentProcessor
from knowledge_bases.kb_manager import KnowledgeBase, KnowledgeBaseRouter
from conversation.memory_manager import ConversationManager
from utils.response_formatter import ResponseFormatter
from tools.compatibility_checker import CompatibilityChecker
from tools.order_status_lookup import OrderStatusLookup

from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType

st.set_page_config(page_title="Enterprise Customer Support", layout="wide")

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory_manager" not in st.session_state:
        st.session_state.memory_manager = ConversationManager()
    if "kb_router" not in st.session_state:
        st.session_state.kb_router = None
    if "formatter" not in st.session_state:
        st.session_state.formatter = ResponseFormatter(
            company_name="TechCorp",
            support_email="support@techcorp.com",
            support_phone="1-800-TECHCORP"
        )
    if "agent" not in st.session_state:
        st.session_state.agent = None

def main():
    st.title("Enterprise Customer Support System")
    
    initialize_session_state()
    
    with st.sidebar:
        st.header("Configuration")
            
        st.header("System Setup")
        st.info("Upload documents to build specialized product knowledge bases. Include 'cloud', 'hardware', or 'software' in the filename to automatically categorize them into distinct knowledge bases.")
        
        uploaded_files = st.file_uploader("Upload Product Documents (PDF, TXT, CSV)", accept_multiple_files=True)
        
        if st.button("Initialize System"):
            if not uploaded_files:
                st.error("Please upload documents first.")
            else:
                with st.spinner("Processing documents and configuring the system..."):
                    doc_processor = DocumentProcessor()
                    kbs = {}
                    
                    os.makedirs("temp_docs", exist_ok=True)
                    for file in uploaded_files:
                        file_path = os.path.join("temp_docs", file.name)
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                            
                        docs = doc_processor.load_documents(file_path)
                        
                        product_category = "General"
                        filename_lower = file.name.lower()
                        if "cloud" in filename_lower:
                            product_category = "Cloud Services"
                        elif "hardware" in filename_lower:
                            product_category = "Hardware Products"
                        elif "software" in filename_lower:
                            product_category = "Software Applications"
                            
                        if product_category not in kbs:
                            kbs[product_category] = []
                        kbs[product_category].extend(docs)
                        
                    knowledge_bases_dict = {}
                    for category, docs in kbs.items():
                        vs = doc_processor.create_vector_store(docs)
                        knowledge_bases_dict[category] = KnowledgeBase(vs)
                        
                    st.session_state.kb_router = KnowledgeBaseRouter(knowledge_bases_dict)
                    
                    llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)
                    tools = [CompatibilityChecker(), OrderStatusLookup()]
                    st.session_state.agent = initialize_agent(
                        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True
                    )
                    
                    st.success("System fully initialized and ready.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("How can I assist you today?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not st.session_state.kb_router or not st.session_state.agent:
                response = "The system is currently offline. Please configure the knowledge bases via the sidebar first."
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                return
                
            with st.spinner("Analyzing request..."):
                try:
                    formatter = st.session_state.formatter
                    lower_prompt = prompt.lower()
                    
                    # Direct queries to agent if they mention orders or compatibility
                    if any(keyword in lower_prompt for keyword in ["compatible", "compatibility", "order", "status"]):
                        raw_response = st.session_state.agent.invoke({"input": prompt})["output"]
                        final_response = formatter.format_response(raw_response)
                    else:
                        raw_response = st.session_state.kb_router.route_query(prompt)
                        final_response = formatter.format_response(raw_response)
                        
                    st.markdown(final_response)
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                    
                    st.session_state.memory_manager.add_user_message(prompt)
                    st.session_state.memory_manager.add_ai_message(final_response)
                    
                except Exception as e:
                    error_msg = formatter.format_error_response(str(e))
                    st.error(error_msg)
                    if "rate limit" in str(e).lower() or "token" in str(e).lower():
                        st.info(formatter.format_escalation_response())

if __name__ == "__main__":
    main()
