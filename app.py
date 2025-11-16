import streamlit as st
import os
import requests
import streamlit.components.v1 as components
from langchain_google_genai import ChatGoogleGenerativeAI
from huggingface_hub import InferenceClient
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- 1. Load API Keys ---
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    HF_TOKEN = st.secrets["HF_API_KEY"]
    if not HF_TOKEN or not os.environ["GOOGLE_API_KEY"]:
        raise KeyError
except KeyError:
    st.error("API Key(s) not found! Please add GOOGLE_API_KEY and HF_API_KEY to your .streamlit/secrets.toml file.")
    st.stop()

# --- 2. Initialize Models & Clients ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
hf_client = InferenceClient(token=HF_TOKEN)

# --- RAG Setup ---
DB_DIR = "db"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
try:
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
except Exception as e:
    st.error(f"Failed to load the knowledge base: {e}")
    st.error("Please run 'python build_db.py' to create the database.")
    st.stop()

retriever = db.as_retriever()
print("RAG 'Handbook' loaded successfully.")

# --- Toxicity Check Function ---
def is_toxic(text_to_check: str) -> bool:
    """Checks text for toxicity using the HF Inference Client."""
    try:
        result = hf_client.text_classification(text_to_check, model="unitary/toxic-bert")
        for label in result:
            if label.label == 'toxic' and label.score > 0.8:
                return True
        return False
    except Exception as e:
        print(f"HF Client Error: {e}")
        return False

# --- 3. App Title ---
st.title("BettrMe.AI Support Bot 🤖")

# --- 4. Initialize All Session Memory (CHANGED) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I am your AI assistant. How can I help you today?"
    })
if "strike_count" not in st.session_state:
    st.session_state.strike_count = 0
if "has_been_forgiven" not in st.session_state:
    st.session_state.has_been_forgiven = False
if "session_lock" not in st.session_state:
    st.session_state.session_lock = False
# --- NEW ---
if "lock_menu_state" not in st.session_state:
    st.session_state.lock_menu_state = "default"


# --- 6. The Final UI & Logic Block --

if st.session_state.session_lock:
    # --- Path 1: CHAT IS LOCKED (New "Other Query" Menu) ---
    st.markdown("---")
    st.warning("This chat is locked. Please select an option for help:")

    # --- MENU ROUTER ---
    if st.session_state.lock_menu_state == "default":
        # --- Main Menu ---
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Help with my Account"):
                st.session_state.lock_menu_state = "account_help"
                st.rerun()
        with col2:
            if st.button("Billing Inquiry"):
                st.session_state.lock_menu_state = "billing_help"
                st.rerun()
        with col3:
            if st.button("Other Query"):
          
                st.session_state.lock_menu_state = "other_query"
                st.rerun()
    
    elif st.session_state.lock_menu_state == "account_help":
        # --- Account Sub-Menu ---
        st.info(
            "**Here's how to get account help:**\n\n"
            "- To reset your password, please click 'Forgot Password' on the login page.\n\n"
            "- To delete your account, go to 'Settings > Profile > Delete Account'."
        )
        if st.button("⬅️ Go Back to Menu"):
            st.session_state.lock_menu_state = "default"
            st.rerun()

    elif st.session_state.lock_menu_state == "billing_help":
        # --- Billing Sub-Menu ---
        st.info(
            "**For Billing Inquiries:**\n\n"
            "Please email our support team directly at `billing@bettrrme.ai` with your account details."
        )
        if st.button("⬅️ Go Back to Menu"):
            st.session_state.lock_menu_state = "default"
            st.rerun()

    elif st.session_state.lock_menu_state == "other_query":
        # --- Other Query Sub-Menu ---
        st.info("Please provide your contact details, and a human agent will get in touch.")
        
        phone_number = st.text_input("Enter your phone number:")
        
        if st.button("Submit Contact Info"):
            if phone_number:
               
                with open("contact_logs.txt", "a") as f:
                    from datetime import datetime
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                    f.write(f"[{timestamp}] New contact request: {phone_number}\n")
                
                st.success("Thank you! Your details are logged. An agent will contact you shortly.")
                st.session_state.lock_menu_state = "default"
                import time
                time.sleep(2)
                st.rerun()
            else:
                st.error("Please enter a phone number.")
        
        if st.button("⬅️ Go Back to Menu"):
            st.session_state.lock_menu_state = "default"
            st.rerun()

else:
    # --- Path 2: CHAT IS UNLOCKED  ---

    if prompt := st.chat_input("What's on your mind?"):
        
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # --- Toxicity Router ---
        if is_toxic(prompt):
            
            # "Probation" Check
            if st.session_state.has_been_forgiven:
                response_content = ("I apologize, but I'm not programmed to engage with that "
                                    "level of abusive language. I am locking this chat. "
                                    "Please select one of the options above to get the help you need.")
                st.session_state.session_lock = True
            
            # Standard 5-Strike System
            elif st.session_state.strike_count == 0:
                response_content = ("I understand you're frustrated, but I'm not able to process that "
                                    "kind of language. 😥 Could you please rephrase your question? "
                                    "I'm here to help with BettrMe.AI.")
                st.session_state.strike_count += 1
                
            elif st.session_state.strike_count == 1:
                response_content = ("I'm sorry, but I'm still unable to help when that language is used "
                                    "due to my programming. I'd really like to resolve your issue. "
                                    "How can I help with your BettrMe.AI account?")
                st.session_state.strike_count += 1
                
            elif st.session_state.strike_count in [2, 3, 4]:
                response_content = ("I apologize, but I'm not programmed to engage with that "
                                    "level of abusive language. I am still here to help with BettrMe.AI. "
                                    "If you'd like to continue, please let me know what I can assist you with.")
                st.session_state.strike_count += 1
                
            else:
                response_content = ("I've given several warnings, and I'm still unable to help. "
                                    "For your security and mine, I am locking this chat. "
                                    "Please select one of the options above to get the help you need.")
                st.session_state.session_lock = True

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_content
            })
            
        else:
            # --- Safe Path (with RAG) ---
            
            # 1. "Forgiveness" Check
            if any(keyword in prompt.lower() for keyword in ["sorry", "sry", "my apologies", "my bad", "okay", "aight", "fine"]):
                st.session_state.strike_count = 0
                st.session_state.has_been_forgiven = True
                response_content = "Thank you, I appreciate that. How can I help you with BettrMe.AI?"
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_content
                })

            else:
                # 2. "On-Topic" vs "Off-Topic" Router
                retrieved_docs = retriever.invoke(prompt)
                
                if retrieved_docs:
                    # --- 3. Path B1: ON-TOPIC (RAG) ---
                    rag_template = """
You are 'BetterBot,' a friendly, patient, and empathetic support assistant for BettrMe.AI.
Use the following pieces of context from the BettrMe.AI handbook to answer the user's question.
If the context doesn't have the answer, just say that you're not sure but you'll do your best to help.

CONTEXT:
{context}

USER ASKS:
{question}

YOUR HELPFUL ANSWER:
"""
                    prompt_template = ChatPromptTemplate.from_template(rag_template)
                    
                    def format_context(docs):
                        return "\n\n".join(doc.page_content for doc in docs)

                    rag_chain = (
                        {"context": retriever | format_context, "question": RunnablePassthrough()}
                        | prompt_template
                        | llm
                        | StrOutputParser()
                    )
                    
                    response_content = rag_chain.invoke(prompt)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_content
                    })
                    
                else:
                    # --- 4. Path B2: OFF-TOPIC ---
                    response_content = "I'm best at helping with BettrMe.AI questions. How can I help with your account or our services?"
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_content
                    })
        st.rerun()
# --- 5. Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 7. Auto-scroll Hack ---
components.html(
    """
    <script>
        // Wait for the Streamlit app to finish rendering
        window.addEventListener("load", function() {
            // Find the main app container
            const container = window.parent.document.querySelector(".main .block-container");
            if (container) {
                // Scroll to the bottom of the container
                container.scrollTop = container.scrollHeight;
            }
        });
    </script>
    """,
    height=0, # Make the HTML component invisible
)