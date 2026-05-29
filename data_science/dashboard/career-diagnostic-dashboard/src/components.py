import streamlit as st


def load_css():
    st.markdown("""
    <style>
        /* Base */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Background */
        .stApp {
            background: linear-gradient(135deg, #0d1117 0%, #0f1923 50%, #0d1117 100%);
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f1923 0%, #131c27 100%);
            border-right: 1px solid rgba(99,179,237,0.15);
        }
        [data-testid="stSidebar"] .stMarkdown {
            color: #94a3b8;
        }

        /* KPI Cards */
        .kpi-card {
            background: linear-gradient(135deg, rgba(15,25,40,0.95) 0%, rgba(20,35,55,0.95) 100%);
            border: 1px solid rgba(99,179,237,0.2);
            border-radius: 16px;
            padding: 24px 20px;
            text-align: center;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
            margin-bottom: 12px;
        }
        .kpi-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, #3b82f6, #06b6d4, #8b5cf6);
        }
        .kpi-card:hover {
            border-color: rgba(99,179,237,0.45);
            transform: translateY(-2px);
            box-shadow: 0 8px 32px rgba(59,130,246,0.18);
        }
        .kpi-value {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #60a5fa, #06b6d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1.1;
            margin-bottom: 6px;
        }
        .kpi-label {
            font-size: 0.78rem;
            font-weight: 500;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .kpi-sub {
            font-size: 0.88rem;
            color: #94a3b8;
            margin-top: 4px;
            font-weight: 500;
        }

        /* Section headers */
        .section-header {
            font-size: 1.1rem;
            font-weight: 700;
            color: #e2e8f0;
            margin-bottom: 4px;
            letter-spacing: -0.01em;
        }
        .section-sub {
            font-size: 0.82rem;
            color: #64748b;
            margin-bottom: 16px;
        }

        /* Page title */
        .page-title {
            font-size: 1.9rem;
            font-weight: 800;
            background: linear-gradient(135deg, #60a5fa 0%, #06b6d4 50%, #8b5cf6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2px;
        }
        .page-subtitle {
            font-size: 0.88rem;
            color: #64748b;
            margin-bottom: 20px;
        }

        /* Pill badges */
        .badge-critical { background: rgba(239,68,68,0.15); color: #f87171; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; border: 1px solid rgba(239,68,68,0.3); }
        .badge-important { background: rgba(245,158,11,0.15); color: #fbbf24; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; border: 1px solid rgba(245,158,11,0.3); }
        .badge-supplementary { background: rgba(34,197,94,0.15); color: #4ade80; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; border: 1px solid rgba(34,197,94,0.3); }

        /* Divider */
        .fancy-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(99,179,237,0.3), transparent);
            margin: 20px 0;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(15,25,40,0.8);
            border-radius: 12px;
            padding: 4px;
            gap: 4px;
            border: 1px solid rgba(99,179,237,0.15);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            color: #64748b;
            font-weight: 500;
            padding: 8px 20px;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #3b82f6, #06b6d4) !important;
            color: white !important;
        }

        /* Charts background */
        .chart-container {
            background: rgba(15,25,40,0.7);
            border: 1px solid rgba(99,179,237,0.12);
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 16px;
        }

        /* Selectbox */
        .stSelectbox > div > div {
            background: rgba(15,25,40,0.9);
            border: 1px solid rgba(99,179,237,0.25);
            border-radius: 10px;
            color: #e2e8f0;
        }

        /* Multiselect */
        .stMultiSelect > div > div {
            background: rgba(15,25,40,0.9);
            border: 1px solid rgba(99,179,237,0.25);
            border-radius: 10px;
        }

        /* Text input */
        .stTextInput > div > div > input {
            background: rgba(15,25,40,0.9);
            border: 1px solid rgba(99,179,237,0.25);
            border-radius: 10px;
            color: #e2e8f0;
        }

        /* Metrics */
        [data-testid="metric-container"] {
            background: rgba(15,25,40,0.7);
            border: 1px solid rgba(99,179,237,0.15);
            border-radius: 12px;
            padding: 12px;
        }

        /* Hide streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #0d1117; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #475569; }
    </style>
    """, unsafe_allow_html=True)
    
    def fancy_divider():
        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)


    def section_header(title, subtitle=None):
        st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

        if subtitle:
            st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


    def page_header(title, subtitle):
        st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)


    def kpi_card(value, label, sub=""):
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)