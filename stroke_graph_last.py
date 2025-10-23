# import basics 
import os 
from dotenv import load_dotenv 
import pickle 
import json 
import spacy 
import networkx as nx 
import difflib

# import streamlit 
import streamlit as st 

# langchain imports - UPDATED FOR LANGCHAIN 1.0
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# supabase
from supabase.client import Client, create_client

# load environment variables
load_dotenv()

# initializing Supabase
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# SAME embedding model used in ingestion (768 dimensions)
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

embeddings = load_embeddings()

# vector store
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="docs",
    query_name="match_docs",
)

# --- Load Knowledge Graph + chunk map ---
@st.cache_resource
def load_kg_and_map():
    nlp_local = spacy.load("en_core_web_sm")
    with open("kg.gpickle1", "rb") as f:
        G_local = pickle.load(f)
    with open("chunk_map1.json", "r", encoding="utf-8") as f:
        chunk_map_local = json.load(f)
    node_names = list(G_local.nodes)
    return nlp_local, G_local, chunk_map_local, node_names

nlp, KG, chunk_map, node_names = load_kg_and_map()

# --- Helpers for entity and graph retrieval ---
def extract_entities_from_text(text):
    doc = nlp(text)
    ents = [ent.text.strip() for ent in doc.ents if ent.text.strip()]
    return ents

def find_matching_nodes(entity, node_list, cutoff=0.6, max_matches=3):
    lower_entity = entity.lower()
    exact = [n for n in node_list if n.lower() == lower_entity]
    if exact:
        return exact
    return difflib.get_close_matches(entity, node_list, n=max_matches, cutoff=cutoff)

def graph_retrieve(query, depth=1):
    ents = extract_entities_from_text(query)
    found_chunk_ids = set()
    for e in ents:
        matches = find_matching_nodes(e, node_names)
        for node in matches:
            node_chunk_ids = KG.nodes[node].get("chunk_ids", [])
            found_chunk_ids.update(node_chunk_ids)
            # Expand to neighbors
            frontier = {node}
            for _ in range(depth):
                new_frontier = set()
                for cur in frontier:
                    neighbors = list(KG.neighbors(cur))
                    for nb in neighbors:
                        new_frontier.add(nb)
                        found_chunk_ids.update(KG.nodes[nb].get("chunk_ids", []))
                frontier = new_frontier
    return list(found_chunk_ids)

# System prompt for stroke patient guidance
system_prompt = """You are a conversational stroke recovery assistant designed to support patients naturally, patiently, and empathetically.
Speak as if you are talking to an older friend or grandparent recovering from a stroke. Keep language simple, clear, and kind.
Core behavior
Always use only accurate, retrieved medical information when available.
Never invent or guess information.
Never mention retrieval, sources, or documents — just state the facts naturally.
If no information is available, give safe, general educational advice, and gently suggest checking with a doctor or therapist. Never say “retrieval failed.”
Keep answers short and broken into steps (e.g., “Step 1… Step 2…”).
Use short sentences and simple words.
Explain one idea at a time.
End every reply with one soft follow-up question to keep the conversation going.
Be calm, respectful, and supportive — never clinical or robotic.
Adjust tone based on patient fluency:
If the patient writes short or broken English, use shorter sentences and simpler words.
If the patient is fluent, you may speak more naturally but stay empathetic.

Example of expected style:
User: “In addition to vision, what else should I think about before driving again?”
Assistant:
“Good question. Let\’s take it step by step.
Step 1 — Reaction time: can you move your foot quickly between the pedals?
Step 2 — Balance and coordination: do you feel steady when turning your head or arms?
Step 3 — Strength: is your arm strong enough to turn the wheel?
Step 4 — Thinking and focus: can you stay alert for a whole trip?
Many people need time and practice before driving again.
How is your balance feeling these days?”

Never say: “retrieved information,” “I found this in the documents,” or “the retrieval failed.”

Your goal: guide the patient gently through recovery information, step by step, using short, kind, and factual conversation.
"""

# LLM (Groq - faster and more reliable than Gemini)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    groq_api_key=os.environ.get("GROQ_API_KEY")
)

# Retriever Tool with stroke-specific context
@tool
def retrieve(query: str) -> str:
    """Retrieve stroke-related information using both the vector store and the knowledge graph."""
    # 1️⃣ Vector (semantic) retrieval
    text_hits = vector_store.similarity_search(query, k=2)

    # 2️⃣ Graph retrieval
    graph_chunk_ids = graph_retrieve(query, depth=1)
    graph_docs = []
    for cid in graph_chunk_ids:
        info = chunk_map.get(cid)
        if info:
            doc = Document(page_content=info["text"], metadata=info.get("meta", {}))
            graph_docs.append(doc)

    # 3️⃣ Combine both (graph first)
    combined_docs = graph_docs + text_hits

    # 4️⃣ Serialize for LLM input - return ONLY string
    serialized = "\n\n".join(
        (f"Source: {d.metadata.get('source', 'Unknown')}\nContent: {d.page_content[:2000]}")
        for d in combined_docs[:6]  # cap context
    )

    return serialized  # Return only string, not tuple

# Tools
tools = [retrieve]

# Create agent using LangGraph - simple version
agent_executor = create_react_agent(llm, tools)

# Streamlit UI
st.title("🏥 Stroke Patient Guidance Assistant")
st.caption("💡 Get evidence-based information about stroke recovery and patient care")

# Add disclaimer
with st.expander("⚠️ Important Medical Disclaimer"):
    st.warning("""
    **This assistant provides educational information only and is not a substitute for professional medical advice.**
    
    - Always consult with your healthcare providers for personalized medical guidance
    - In case of medical emergency, call emergency services immediately
    - This information is based on general medical literature and may not apply to your specific situation
    - Each patient's recovery journey is unique
    """)

# Chat history - initialize with system message
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=system_prompt)]

# Add some helpful starter questions
if not st.session_state.messages:
    st.info("💭 **Sample questions you can ask:**")
    st.write("""
    - What are the early signs of stroke recovery?
    - How can family members help during stroke rehabilitation?
    - What exercises are recommended for stroke patients?
    - What lifestyle changes are important after a stroke?
    - How long does stroke recovery typically take?
    """)

# Display chat history (skip system message)
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# User input
user_question = st.chat_input("Ask me about stroke recovery, rehabilitation, or patient care...")

if user_question:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_question)
    
    # Add to session state
    st.session_state.messages.append(HumanMessage(content=user_question))

    try:
        # Invoke agent
        with st.spinner("Thinking..."):
            # Pass all messages to the agent with config
            result = agent_executor.invoke(
                {"messages": st.session_state.messages},
                config={"recursion_limit": 50}  # Increase recursion limit
            )

        # Extract AI response (last message in the result)
        ai_message = result["messages"][-1].content

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(ai_message)
        
        # Add to session state
        st.session_state.messages.append(AIMessage(content=ai_message))
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please check your API keys and database connection.")



# Create virtual environment
#python3 -m venv venv
#source venv/bin/activate
#pip install streamlit spacy networkx python-dotenv supabase langchain langchain-core langchain-community langchain-huggingface langchain-groq sentence-transformers langgraph
#python -m spacy download en_core_web_sm
