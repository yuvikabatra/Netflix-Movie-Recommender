"""
auth.py
Username/password authentication for the Netflix Recommendation System app.

Storage: local JSON file (users.json) with salted + hashed passwords
(PBKDF2-HMAC-SHA256, 100k iterations). Swap load_users/save_users for a
real database call if you need multi-instance or production-grade storage.
"""

import json
import os
import hashlib
import hmac
import secrets
import streamlit as st

USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def save_users(users: dict) -> None:
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), 100_000
    ).hex()
    return hashed, salt


def _verify_password(password: str, salt: str, hashed: str) -> bool:
    check, _ = _hash_password(password, salt)
    return hmac.compare_digest(check, hashed)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def signup(username: str, password: str, confirm_password: str) -> tuple[bool, str]:
    username = username.strip()

    if not username or not password:
        return False, "Username and password cannot be empty."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if password != confirm_password:
        return False, "Passwords do not match."

    users = load_users()
    if username in users:
        return False, "That username is already taken."

    hashed, salt = _hash_password(password)
    users[username] = {"hash": hashed, "salt": salt}
    save_users(users)
    return True, "Account created. You can now log in."


def login(username: str, password: str) -> tuple[bool, str]:
    users = load_users()
    record = users.get(username.strip())

    if not record or not _verify_password(password, record["salt"], record["hash"]):
        return False, "Invalid username or password."

    st.session_state["authenticated"] = True
    st.session_state["username"] = username.strip()
    return True, "Logged in successfully."


def logout() -> None:
    st.session_state["authenticated"] = False
    st.session_state["username"] = None


def is_authenticated() -> bool:
    return st.session_state.get("authenticated", False)


def current_user() -> str | None:
    return st.session_state.get("username")


# ---------------------------------------------------------------------------
# UI — styled to match the Netflix theme already defined in app.py
# ---------------------------------------------------------------------------

def _inject_auth_css() -> None:
    st.markdown(
        """
        <style>
        .auth-logo {
            color: #E50914;
            font-weight: 800;
            font-size: 42px;
            letter-spacing: 1px;
            text-align: center;
            margin-bottom: 0px;
        }
        .auth-sub {
            color: #9a9a9a;
            font-size: 13px;
            letter-spacing: 2px;
            text-align: center;
            margin-top: -6px;
            margin-bottom: 28px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def login_form() -> None:
    with st.form("login_form", clear_on_submit=False):
        st.subheader("Log In")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        submitted = st.form_submit_button("Log In", use_container_width=True)

        if submitted:
            success, message = login(username, password)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)


def signup_form() -> None:
    with st.form("signup_form", clear_on_submit=True):
        st.subheader("Sign Up")
        username = st.text_input("Choose a username", key="signup_username")
        password = st.text_input("Choose a password", type="password", key="signup_password")
        confirm = st.text_input("Confirm password", type="password", key="signup_confirm")
        submitted = st.form_submit_button("Sign Up", use_container_width=True)

        if submitted:
            success, message = signup(username, password, confirm)
            if success:
                st.success(message)
            else:
                st.error(message)


def logout_button() -> None:
    st.sidebar.markdown("---")
    st.sidebar.write(f"👤 Logged in as **{current_user()}**")
    if st.sidebar.button("Log Out", use_container_width=True):
        logout()
        st.rerun()


def auth_gate() -> bool:
    """
    Call this once, right after st.set_page_config (and after your CSS,
    if you want the login screen themed too). Renders a centered
    login/signup screen if the user isn't authenticated yet, and returns
    True once they are so the rest of the app can render.
    """
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"]:
        return True

    _inject_auth_css()

    left, mid, right = st.columns([1, 1.2, 1])
    with mid:
        st.markdown('<div class="auth-logo">NETFLIX</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">RECOMMENDATION SYSTEM</div>', unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["Log In", "Sign Up"])
        with tab_login:
            login_form()
        with tab_signup:
            signup_form()

    return False