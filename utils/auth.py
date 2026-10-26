import hashlib
import hmac
import secrets
import sqlite3
from pathlib import Path
import streamlit as st

DB_PATH = Path(__file__).resolve().parent.parent / "users.db"

def init_auth_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt.encode("utf-8"),
        n=16384, r=8, p=1
    )
    return digest.hex(), salt

def verify_password(password, stored_hash, salt):
    calculated, _ = hash_password(password, salt)
    return hmac.compare_digest(calculated, stored_hash)

def create_user(user_id, password):
    user_id = user_id.strip()
    if not user_id or not password:
        return False, "User ID and password are required."
    if len(password) < 6:
        return False, "Password must contain at least 6 characters."
    password_hash, salt = hash_password(password)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO users(user_id, password_hash, salt) VALUES (?, ?, ?)",
                (user_id, password_hash, salt)
            )
            conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "This User ID already exists."

def authenticate(user_id, password):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT password_hash, salt FROM users WHERE user_id = ?",
            (user_id.strip(),)
        ).fetchone()
    if row and verify_password(password, row[0], row[1]):
        return True
    return False

def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def login_screen():
    init_auth_db()
    _, centre, _ = st.columns([1, 1.4, 1])
    centre.empty()
    logo = Path(__file__).resolve().parent.parent / "assets" / "ust_logo.jpg"
    st.markdown('<div class="ust-title">UST Software India Pvt Ltd</div>', unsafe_allow_html=True)
    if logo.exists():
        st.image(str(logo), width=70)
    st.markdown('<div class ="ust-title">Data Analysis Using Streamlit in Python</div>', unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center'>Login</h2>", unsafe_allow_html=True)
    st.caption("Sign in to continue.")

    tab_login, tab_register = st.tabs(["Login", "Create account"])

    with tab_login:
        user_id = st.text_input("User ID", key="login_user")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", type="primary", use_container_width=True):
            if authenticate(user_id, password):
                st.session_state.logged_in = True
                st.session_state.user_id = user_id.strip()
                st.rerun()
            else:
                st.error("Invalid User ID or password.")

    with tab_register:
        new_id = st.text_input("User ID", key="register_user")
        new_password = st.text_input("Password", type="password", key="register_password")
        confirm = st.text_input("Confirm password", type="password", key="register_confirm")
        if st.button("Create account", use_container_width=True):
            if new_password != confirm:
                st.error("Passwords do not match.")
            else:
                ok, message = create_user(new_id, new_password)
                if ok:
                    st.toast("Account created successfully. Continue with login.", icon="✅")
                else:
                    st.error(message)
