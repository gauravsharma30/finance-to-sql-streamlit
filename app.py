import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# ─── CONFIG ────────────────────────────────────────────────────────────
GLOSSARY_CSV = "glossary_v1.csv"
GROUPING_CSV = "fbi_grouping_master.csv"
EMBED_MODEL  = "sentence-transformers/all-mpnet-base-v2"

# Defaults
DEFAULT_ENTITY_ID = 6450
DEFAULT_TAXONOMY  = 71
DEFAULT_CURRENCY  = "INR"

# ─── CACHING ────────────────────────────────────────────────────────────
@st.cache_resource
def load_data_and_models():
    gloss_df = pd.read_csv(GLOSSARY_CSV)
    group_df = pd.read_csv(GROUPING_CSV)
    model    = SentenceTransformer(EMBED_MODEL)

    gloss_terms = gloss_df["Glossary"].tolist()
    gloss_embs  = model.encode(gloss_terms, convert_to_tensor=True)

    group_lbls  = group_df["grouping_label"].tolist()
    group_embs  = model.encode(group_lbls, convert_to_tensor=True)

    return gloss_df, group_df, model, gloss_terms, gloss_embs, group_lbls, group_embs

gloss_df, group_df, model, gloss_terms, gloss_embs, group_lbls, group_embs = load_data_and_models()

# ─── WORKFLOW FUNCTIONS ─────────────────────────────────────────────────
def extract_glossary(nl: str) -> str:
    q_emb = model.encode(nl, convert_to_tensor=True)
    sims  = util.cos_sim(q_emb, gloss_embs)[0]
    return gloss_terms[int(torch.argmax(sims))]

def lookup_grouping(term: str):
    q_emb = model.encode(term, convert_to_tensor=True)
    sims  = util.cos_sim(q_emb, group_embs)[0]
    idx  = int(torch.argmax(sims))
    return group_df.loc[idx, "grouping_id"], group_lbls[idx]

def resolve_period(nl: str) -> str:
    # Stub: you can replace this with your real logic
    return "2024_M_PRD_3"

def infer_scenario(nl: str) -> str:
    return "Cashflow" if "cash" in nl.lower() else "Actual"

def infer_nature(nl: str) -> str:
    return "Standalone"

SQL_TMPL = """
SELECT value
FROM fbi_entity_analysis_report
WHERE entity_id            = {entity_id}
  AND grouping_id          = {grouping_id}
  AND period_id            = '{period_id}'
  AND nature_of_report     = '{nature}'
  AND scenario             = '{scenario}'
  AND taxonomy_id          = {taxonomy}
  AND reporting_currency   = '{currency}'
;
"""
def make_sql(params: dict) -> str:
    return SQL_TMPL.format(**params)

# ─── STREAMLIT UI ───────────────────────────────────────────────────────
st.title("📊 Finance-to-SQL Demo")

nl_query = st.text_input("Enter your finance question:")

if st.button("Generate SQL") and nl_query:
    # Run the pipeline
    gloss, (gid, glabel) = extract_glossary(nl_query), lookup_grouping(extract_glossary(nl_query))
    period   = resolve_period(nl_query)
    scenario = infer_scenario(nl_query)
    nature   = infer_nature(nl_query)

    params = {
        "entity_id":   DEFAULT_ENTITY_ID,
        "grouping_id": gid,
        "period_id":   period,
        "nature":      nature,
        "scenario":    scenario,
        "taxonomy":    DEFAULT_TAXONOMY,
        "currency":    DEFAULT_CURRENCY
    }
    sql = make_sql(params)

    # Display results
    st.subheader("🔍 Mapping Results")
    st.write("**Glossary Term:**", gloss)
    st.write("**Grouping Label:**", glabel)
    st.write("**Grouping ID:**", gid)
    st.write("**Period ID:**", period)
    st.write("**Scenario:**", scenario)
    st.write("**Nature:**", nature)

    st.subheader("🛠 Generated SQL")
    st.code(sql, language="sql")
