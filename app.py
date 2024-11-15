import os
import streamlit as st
from streamlit_navigation_bar import st_navbar
import pages as pg

st.set_page_config(initial_sidebar_state="collapsed")

pages = ["Install", "User Guide", "API", "Examples", "Community", "GitHub"]
parent_dir = os.path.dirname(os.path.abspath(_file_))
logo_path = os.path.join(parent_dir, "cubes.svg")
urls = {"GitHub": "https://github.com/gabrieltempass/streamlit-navigation-bar"}
styles = {
    "nav": {
        "background-color": "royalblue",
        "justify-content": "left",
    },
    "img": {
        "padding-right": "14px",
    },
    "span": {
        "color": "white",
        "padding": "14px",
    },
    "active": {
        "color": "var(--text-color)",
        "background-color": "white",
        "font-weight": "normal",
        "padding": "14px",
    }
}

page = st_navbar(
    pages,
    logo_path=logo_path,
    urls=urls,
    styles=styles,
    options=False,
)

functions = {
    "Home": pg.show_home,
    "Install": pg.show_install,
    "User Guide": pg.show_user_guide,
    "API": pg.show_api,
    "Examples": pg.show_examples,
    "Community": pg.show_community,
}
go_to = functions.get(page)
if go_to:
    go_to()
