import streamlit as st
import cohere
import json
import subprocess
import os
import numpy as np
from chromadb import PersistentClient
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings
import time

co = cohere.Client("UwaEhrCG2fnBwdNvkS3TqKVhxgBck5M600evNkmz")
st.set_page_config(page_title="3GPP Query Assistant", layout="wide")

BASE_DIR = os.path.dirname(__file__)
CHROMA_PATH = os.path.join(BASE_DIR, "../chroma_storage") 
client = PersistentClient(path=CHROMA_PATH)

WVER2_PATH = os.path.join(BASE_DIR, "wver2.py")
EMBED2_PATH = os.path.join(BASE_DIR, "embed2.py")

if "users" not in [c.name for c in client.list_collections()]:
    client.create_collection("users")
users_collection = client.get_collection("users")

def signup(username, password):
    user_data = users_collection.get(where={"username": username})
    if user_data["ids"]:
        return False
    users_collection.add(
        documents=[password],
        metadatas=[{"username": username}],
        ids=[username]
    )
    return True

def login(username, password):
    user_data = users_collection.get(where={"username": username})
    if not user_data["ids"]:
        return False
    stored_password = user_data["documents"][0]
    return password == stored_password

def load_user_chats(username):
    chats = {}
    titles = {}
    if username not in [c.name for c in client.list_collections()]:
        client.create_collection(username)
        return chats, titles

    user_chat_collection = client.get_collection(username)
    data = user_chat_collection.get(include=["metadatas", "documents"])
    for doc, meta in zip(data["documents"], data["metadatas"]):
        chat_id = meta["chat_id"]
        title = meta["title"]
        messages = json.loads(doc)
        chats[chat_id] = messages
        titles[chat_id] = title
    return chats, titles

def save_chat(username, chat_id, chat_title, messages):
    user_chat_collection = client.get_collection(username)
    serialized = json.dumps(messages)
    user_chat_collection.upsert(
        documents=[serialized],
        ids=[chat_id],
        metadatas=[{"chat_id": chat_id, "title": chat_title}]
    )

def delete_chat(username, chat_id):
    if username in [c.name for c in client.list_collections()]:
        user_chat_collection = client.get_collection(username)
        user_chat_collection.delete(ids=[chat_id])

def truncate_text(text, max_tokens=1000):
    tokens = text.split()
    return " ".join(tokens[:max_tokens])

def batch_cosine_similarity(query_emb, db_embeds):
    return cosine_similarity([query_emb], db_embeds)[0]

def get_top_k_chunks(user_input, collection, k=10):
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    truncated_query = truncate_text(user_input, 1000)
    query_embedding = embed_model.embed_query(truncated_query)

    data = collection.get(include=["embeddings", "documents"])
    embeddings = np.array(data["embeddings"])
    documents = data["documents"]

    scores = batch_cosine_similarity(query_embedding, embeddings)
    top_indices = np.argsort(scores)[::-1][:k]
    top_chunks = [documents[i] for i in top_indices]

    return top_chunks

def truncate_to_token_limit(text, limit=3500):
    tokens = text.split()
    return " ".join(tokens[:limit])

for key in ["username", "logged_in", "show_signup", "chat_history", "all_chats", "active_chat_id", "chat_titles"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "username" else False if key in ["logged_in", "show_signup"] else {}

def show_login():
    st.title("3GPP Assistant Login")
    login_username = st.text_input("Username")
    login_password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login(login_username, login_password):
            st.success("Logged in!")
            st.session_state.logged_in = True
            st.session_state.username = login_username
            chats, titles = load_user_chats(login_username)
            st.session_state.all_chats = chats
            st.session_state.chat_titles = titles
            st.session_state.active_chat_id = None 
            st.rerun()
        else:
            st.error("Invalid credentials")
    if st.button("Sign Up"):
        st.session_state.show_signup = True

def show_signup():
    st.title("Sign Up for 3GPP Assistant")
    new_username = st.text_input("Choose a username")
    new_password = st.text_input("Choose a password", type="password")
    if st.button("Create Account"):
        if signup(new_username, new_password):
            st.success("Account created. Please log in.")
            st.session_state.show_signup = False
        else:
            st.error("Username already exists.")
    if st.button("Go to Login"):
        st.session_state.show_signup = False

if not st.session_state.logged_in:
    if st.session_state.show_signup:
        show_signup()
    else:
        show_login()
    st.stop()

st.sidebar.title("3GPP Chat Interface")
st.sidebar.markdown(f"**👤 User:** {st.session_state.username}")

if st.sidebar.button("New Chat"):
    new_id = f"chat_{int(time.time())}"
    st.session_state.active_chat_id = new_id
    st.session_state.all_chats[new_id] = []
    st.session_state.chat_titles[new_id] = "Untitled Chat"
    st.rerun()

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = None
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("Chats")
for chat_id in list(st.session_state.all_chats.keys())[::-1]:
    current_title = st.session_state.chat_titles.get(chat_id, "Untitled Chat")
    col1, col2 = st.sidebar.columns([0.8, 0.2])
    if col1.button(current_title, key=chat_id):
        st.session_state.active_chat_id = chat_id
        st.rerun()
    if col2.button("🗑️", key=f"delete_{chat_id}"):
        delete_chat(st.session_state.username, chat_id)  
        del st.session_state.all_chats[chat_id]
        del st.session_state.chat_titles[chat_id]
        if chat_id == st.session_state.active_chat_id:
            st.session_state.active_chat_id = None
        st.rerun()

with st.sidebar.expander("Rename Current Chat"):
    if st.session_state.active_chat_id:
        new_title = st.text_input("Enter new title", value=st.session_state.chat_titles.get(st.session_state.active_chat_id, "Untitled Chat"))
        if st.button("Save Title"):
            st.session_state.chat_titles[st.session_state.active_chat_id] = new_title
            save_chat(
                st.session_state.username,
                st.session_state.active_chat_id,
                new_title,
                st.session_state.all_chats[st.session_state.active_chat_id]
            )

st.title("3GPP Query Assistant")

if st.session_state.active_chat_id:
    active_chat = st.session_state.all_chats[st.session_state.active_chat_id]

    for msg in active_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about 3GPP...")
    if user_input:
        st.chat_message("user").markdown(user_input)
        active_chat.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Fetching response..."):
                task_prompt = f"""
You are a telecom expert. Your job is to analyze a user's query about 3GPP and respond in the following strict JSON-like format:

{{ 
  "task": "comparison" or "summary", 
  "releases": ["reX", "reY"] or ["reZ"] 
}}

Rules:
- If the user asks to compare two releases (e.g., release 12 and release 13), return task as "comparison" and releases as ["re12", "re13"].
- If the user asks to summarize one release, return task as "summary" and releases as ["reZ"].
- Do NOT add any explanation or extra text outside the JSON-like format.

User Query: \"\"\"{user_input}\"\"\"
"""
                print("Sending task prompt to LLM")
                try:
                    response = co.chat(
                        model="command-r",
                        message=task_prompt,
                        temperature=0.2,
                        max_tokens=100
                    )
                    parsed = json.loads(response.text.strip().replace("'", '"'))
                    task = parsed["task"]
                    releases = parsed["releases"]
                    print(f"Parsed Task: {task} | Releases: {releases}")
                except:
                    st.error("Failed to interpret the task.")
                    st.stop()

                try:
                    if task == "comparison" and len(releases) == 2:
                        version1, version2 = sorted(releases, key=lambda x: int(x[2:]))
                        coll_name = f"{version1}_vs_{version2}"
                        if coll_name not in [c.name for c in client.list_collections()]:
                            env = os.environ.copy()
                            env["VERSION1"] = version1
                            env["VERSION2"] = version2
                            env["CHROMA_PATH"] = CHROMA_PATH
                            print(f"Running comparison subprocess: {WVER2_PATH}")
                            subprocess.run(["python", WVER2_PATH], env=env, check=True)
                        collection = client.get_collection(coll_name)

                    elif task == "summary" and len(releases) == 1:
                        version = releases[0]
                        if version not in [c.name for c in client.list_collections()]:
                            env = os.environ.copy()
                            env["CHROMA_PATH"] = CHROMA_PATH
                            print(f"Running summary subprocess: {EMBED2_PATH}")
                            subprocess.run(["python", EMBED2_PATH], env=env, check=True)
                        collection = client.get_collection(version)

                    else:
                        st.warning("Invalid query format. Use summary or comparison.")
                        st.stop()

                    top_chunks = get_top_k_chunks(user_input, collection)
                    context = truncate_to_token_limit("\n\n".join(top_chunks))

                    final_prompt = f"""
You are a telecom expert AI. Based on the following retrieved information, respond to the user query in detail.

User Query: \"\"\"{user_input}\"\"\"

Relevant Information: \"\"\"{context}\"\"\"
"""
                    print("Sending final prompt to LLM")
                    reply = co.chat(
                        model="command-r",
                        message=final_prompt,
                        temperature=0.5,
                        max_tokens=1000
                    ).text.strip()

                    st.markdown(reply)
                    active_chat.append({"role": "assistant", "content": reply})

                    if len(active_chat) == 2:
                        first_msg = user_input[:30]
                        title = first_msg + "..." if len(user_input) > 30 else first_msg
                        st.session_state.chat_titles[st.session_state.active_chat_id] = title

                    save_chat(
                        st.session_state.username,
                        st.session_state.active_chat_id,
                        st.session_state.chat_titles[st.session_state.active_chat_id],
                        active_chat
                    )

                except Exception as e:
                    st.error(f"Processing error: {e}")
else:
    st.info("👈 Select or create a chat to get started.")