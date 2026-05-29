import streamlit as st
import pandas as pd


@st.cache_data
def load_data():
    edges = pd.read_csv("data/skill_network_edges.csv")
    kb = pd.read_csv("data/knowledge_base_skills.csv")
    jobs = pd.read_csv("data/master_job_postings.csv")

    jobs["market_source"] = jobs["market_source"].fillna("Global")

    return edges, kb, jobs