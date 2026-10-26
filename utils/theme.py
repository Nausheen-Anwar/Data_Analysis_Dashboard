from pathlib import Path

import streamlit as st

from utils.palette import css_vars

CSS_PATH = Path(__file__).resolve().parent.parent / "static" / "css" / "dashboard.css"
CSS = CSS_PATH.read_text(encoding="utf-8")


def inject_css(logged_in: bool = True):
    st.markdown(f"<style>\n{css_vars()}\n{CSS}\n</style>", unsafe_allow_html=True)


def banner(title: str, text: str):
    st.markdown(f'<div class="lab-banner"><h2>{title}</h2><p>{text}</p></div>',
                unsafe_allow_html=True)


def concept(text: str):
    st.markdown(f'<div class="concept">{text}</div>', unsafe_allow_html=True)


def section_banner(title: str, text: str):
    banner(title, text)
