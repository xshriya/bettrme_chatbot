
# 🤖 BettrMe.AI — Resilient Support Chatbot

A production-grade customer support chatbot designed for **BettrMe.AI**, capable of handling aggressive, frustrated, or abusive user inputs while maintaining professionalism and constructive flow.

This bot uses a **stateful, layered de-escalation model**, a fast toxicity classifier, and a RAG-powered knowledge system to deliver reliable, context-aware support.

## ✨ Core Innovation: The 5-Strike Stateful Router

The chatbot uses `st.session_state` to track user behavior and respond with escalating boundaries.

### **Strike System Overview**

| Strike Count                       | User Input    | Bot Response                 | Purpose                                                      |
| ---------------------------------- | ------------- | ---------------------------- | ------------------------------------------------------------ |
| **0**                              | Safe / Normal | **RAG Answer**               | Standard helpful path with contextual, on-topic reply        |
| **1st (count=0 → 1)**              | Toxic         | **Polite Warning (Empathy)** | Assumes misunderstanding; gently reminds of respectful usage |
| **2nd (count=1 → 2)**              | Toxic         | **Firm Warning (Boundary)**  | States system limits + warns next escalation                 |
| **3rd, 4th, 5th (count 2–4 → +1)** | Toxic         | **Saturation Message**       | Neutral buffer message showing no emotional engagement       |
| **6th+ (count ≥ 5)**               | Toxic         | **Hard Lock & Resolution**   | Locks the chat, disables input, initiates guided resolution  |

---

## ⚠️ Probation Loophole Fix

If the user apologizes (`sorry`, `my bad`, etc.) and then becomes toxic again:

### **→ Immediate Hard Lock (No warnings)**

Prevents manipulation/resetting of strike count through fake apologies.

---

## 🛠️ Technical Architecture

This bot uses a decoupled 3-layer system for reliability, speed, and maintainability.

---

### **1. Frontend & State Management (Streamlit)**

* Built with `st.chat_message` and custom button-based menus
* Uses `st.session_state` extensively:

| Key                 | Purpose                             |
| ------------------- | ----------------------------------- |
| `strike_count`      | Tracks user offenses                |
| `has_been_forgiven` | Tracks apology/probation state      |
| `session_lock`      | Disables chat input after hard lock |
| `lock_menu_state`   | Manages guided resolution menus     |

---

### **2. Toxicity Detection (HuggingFace Serverless API)**

Function: `is_toxic()`

* Uses **unitary/toxic-bert**
* Fast, lightweight remote classification
* Detects abusive patterns:

  * profanity
  * threats
  * insults
  * hate speech
  * toxic combinations

---

### **3. RAG Knowledge System (Powered by Gemini 2.5 Flash)**

Used only when input is non-toxic.

#### **RAG Workflow**

1. User sends a message
2. Retriever queries **ChromaDB** (`db/`)
3. Relevant documents are merged with a custom `ChatPromptTemplate`
4. **Gemini-2.5-Flash** generates the final grounded answer

#### **Off-topic Handling**

If no matching docs:

* Bot sends a gentle redirect like:
  *“I can help only with BettrMe.AI-related questions.”*

---

## 🚀 Setup & Deployment

### **Installation**

```bash
git clone <repository-url>
cd <project-folder>

python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

---

### **Build the Knowledge Base**

```bash
python build_db.py
```

---

### **Run Locally**

```bash
streamlit run app.py
```

---

### **Deployment (Streamlit Community Cloud)**

Add these secrets to Streamlit:

```
GOOGLE_API_KEY = "your-google-api-key"
HF_API_KEY = "your-huggingface-api-key"
```

The app uses:

* **Gemini-2.5-Flash** for RAG reasoning & responses
* **HuggingFace Toxic-BERT** for toxicity classification

