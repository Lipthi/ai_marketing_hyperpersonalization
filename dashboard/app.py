# dashboard/app.py

import os, sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import streamlit as st
import pandas as pd
import json
import time
from pathlib import Path
import requests

from backend.generate_messages import run_generation_stream
from backend.rule_engine import generate_bulk_offers
from backend.propensity_model import train_and_save_model

if "gen_running" not in st.session_state:
    st.session_state.gen_running = False

if "last_rendered_count" not in st.session_state:
    st.session_state.last_rendered_count = 0

# -----------------------------
# Helper Functions
# -----------------------------

DATA_PATH = "../data/processed/messages.json"
CUSTOMER_PATH = "../data/processed/customers.json"
LOG_PATH = "../backend/logs/progress.json"

def get_dataframe(messages: list) -> pd.DataFrame:
    if not messages:
        return pd.DataFrame()

    df = pd.DataFrame(messages)

    # Ensure required columns exist for table display
    for col, default in [
        ("customer_id", ""),
        ("name", ""),
        ("loyalty_status", "Unknown"),
        ("engagement_score", 0.0),
        ("propensity_score", 0.0),
        ("discount_pct", 0),
        ("clicked", 0),
        ("converted", 0),
        ("offer_message_plain", "")
    ]:
        if col not in df.columns:
            df[col] = default

    # Clean up text column
    df["offer_message_plain"] = df["offer_message_plain"].fillna("").astype(str).str.strip()

    # Sort newest first
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp", ascending=False)

    return df


def load_messages():
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r") as f:
            msgs = json.load(f)
        # Ensure every message has offer_message_plain
        changed = False
        for m in msgs:
            if "offer_message_plain" not in m:
                m["offer_message_plain"] = (m.get("final_message") or m.get("offer_message") or "").strip()
                changed = True
            m.setdefault("clicked", 0)
            m.setdefault("converted", 0)
        if changed:
            with open(DATA_PATH, "w") as f:
                json.dump(msgs, f, indent=2)
        return msgs
    return []


def save_messages(data):
    with open(DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)

def load_customers_from_file(p: str):
    if os.path.exists(p):
        try:
            with open(p, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def fav_product_from_tx(recent_transactions):
    # safe favourite product extraction
    if not recent_transactions:
        return ""
    try:
        prods = [t.get("product") for t in recent_transactions if isinstance(t, dict) and t.get("product")]
        if not prods:
            return ""
        return max(set(prods), key=prods.count)
    except Exception:
        return ""

def preview_dataframe_for_upload(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a compact preview dataframe for uploaded customers.
    Columns: customer_id, name, loyalty_status, engagement_score, num_transactions, fav_product
    """

    tmp = df.copy()

    # Ensure needed columns exist 
    tmp["customer_id"] = tmp.get("customer_id") if "customer_id" in tmp.columns else tmp.index.astype(str)
    tmp["name"] = tmp.get("name", "")
    tmp["loyalty_status"] = tmp.get("loyalty_status", "Bronze")
    tmp["engagement_score"] = pd.to_numeric(tmp.get("engagement_score", 0.0), errors="coerce").fillna(0.0)

    # recent_transactions may be list - count them and find fav product
    def _num_tx(x):
        try:
            return len(x) if isinstance(x, list) else 0
        except Exception:
            return 0

    tmp["num_transactions"] = tmp.get("recent_transactions").apply(_num_tx) if "recent_transactions" in tmp.columns else 0
    tmp["fav_product"] = tmp.get("recent_transactions").apply(lambda x: fav_product_from_tx(x) if isinstance(x, list) else "")

    preview_cols = ["customer_id", "name", "loyalty_status", "engagement_score", "num_transactions", "fav_product"]
    for c in preview_cols:
        if c not in tmp.columns:
            tmp[c] = None

    return tmp[preview_cols]

# -----------------------------
# Streamlit Layout
# -----------------------------

st.set_page_config(
    page_title="Hyperpersonalization Control Center",
    layout="wide"
)

st.title("Hyperpersonalization Control Center")

messages = load_messages()


# Sidebar controls
st.sidebar.header("🧭 Campaign Controls")

generate_btn = st.sidebar.button("Generate Offers")
retrain_btn = st.sidebar.button("Retrain Model")

show_whatsapp_panel = st.sidebar.checkbox("Show WhatsApp panel", value=False, help="Open panel to send test WhatsApp messages")

st.sidebar.divider()
st.sidebar.subheader("Filters")

loyalty_filter = st.sidebar.selectbox(
    "Loyalty Tier", ["All", "Bronze", "Silver", "Gold", "Platinum"]
)

engagement_range = st.sidebar.slider(
    "Engagement Score Range", 0.0, 1.0, (0.0, 1.0), 0.05
)

churn_filter = st.sidebar.selectbox("Churned / Active", ["All", "Churned", "Active"])

st.sidebar.divider()
uploaded_file = st.sidebar.file_uploader("Upload New Customer Batch (CSV)", type=["json","csv"])

if "customer_data" not in st.session_state:
    st.session_state.customer_data = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".json"):
            uploaded_json = json.load(uploaded_file)
            df_uploaded = pd.DataFrame(uploaded_json)
        else:
            df_uploaded = pd.read_csv(uploaded_file)

        st.session_state.customer_data = df_uploaded

        st.sidebar.success("Customer data loaded! ")
        st.write("Uploaded Preview:")
        st.dataframe(preview_dataframe_for_upload(df_uploaded), width="stretch", hide_index=True)

    except Exception as e:
        st.sidebar.error(f"Failed to read file: {e}")
        st.stop()
else:
    st.sidebar.info("Upload customers (JSON file) to generate new offers ⬆️")


# -----------------------------
# Actions
# -----------------------------
if retrain_btn:
    with st.spinner("Training model..."):
        model = train_and_save_model()
    st.success("✅ Model retrained and saved successfully!")

if generate_btn:
    if st.session_state.customer_data is None:
        st.warning("Please upload a customer dataset first!")
        st.stop()

    customers = st.session_state.customer_data.to_dict(orient="records")

    # Reset UI
    st.session_state.gen_running = True
    st.session_state.last_rendered_count = 0

    # UI placeholders
    progress_bar = st.progress(0.0, text="Starting generation...")
    table_placeholder = st.empty()

    messages_live = []

    def on_offer(offer, done, total):
        messages_live.append(offer)

        # Persist to disk (WhatsApp depends on this)
        save_messages(messages_live)

        # Update UI immediately
        df_live = get_dataframe(messages_live)
        table_placeholder.dataframe(df_live, width="stretch", hide_index=True)

        progress_bar.progress(done / total, text=f"{done}/{total} - {offer.get('name')}")

    # RUN INSIDE STREAMLIT
    run_generation_stream(customers, on_offer)

    st.session_state.gen_running = False
    st.success("🎉 All offers generated live!")

# -----------------------------
# WhatsApp test panel (persistent)
# -----------------------------
if show_whatsapp_panel:
    st.subheader("Send WhatsApp Message.")

    # reload messages fresh so UI always shows latest generated messages
    messages = load_messages()

    if not messages:
        st.info("No generated messages found — generate offers first.")
    else:
        # build list of customer ids safely
        customer_ids = sorted([m.get("customer_id") for m in messages if m.get("customer_id")])
        if not customer_ids:
            st.info("No customer ids found in messages.json")
        else:
            selected_customer = st.selectbox("Select a customer to send WhatsApp messages", customer_ids)

            if st.button("Send Now"):
                with st.spinner("Sending WhatsApp Message..."):
                    try:
                        url = f"http://127.0.0.1:8000/send-whatsapp/{selected_customer}"
                        response = requests.get(url, timeout=15)
                        # robustly parse JSON
                        try:
                            result = response.json()
                        except Exception:
                            result = {"error": "Backend did not return JSON", "raw_text": response.text}

                        st.json(result)

                        # handle several possible response key names
                        whatsapp_payload = result.get("whatsapp_result") or result.get("whatsapp_api_result") or result.get("whatsapp_result", {})
                        status = None
                        if isinstance(whatsapp_payload, dict):
                            status = whatsapp_payload.get("status")
                        # success conditions: explicit "sent" or message_sid present
                        if status == "sent" or (isinstance(whatsapp_payload, dict) and whatsapp_payload.get("message_sid")):
                            st.success("WhatsApp message sent successfully!")
                        else:
                            st.error(f"Failed to send message. Response: {result}")
                    except Exception as e:
                        st.error(f"Request failed : {e}")
            
# -----------------------------
# Display Offers 
# -----------------------------
st.subheader("Generated Offers")

messages = load_messages()
df = get_dataframe(messages)

if not df.empty:
    if loyalty_filter != "All":
        df = df[df["loyalty_status"] == loyalty_filter]

    df = df[df["engagement_score"].between(*engagement_range)]

    if churn_filter == "Churned":
        df = df[df["churn"] == 1]
    elif churn_filter == "Active":
        df = df[df["churn"] == 0]

    # -----------------
    # Select only key columns for final display
    # -----------------
    show_cols = [
        "customer_id", "offer_id", "name", "loyalty_status",
        "churn", "engagement_score", "propensity_score",
        "discount_pct", "clicked", "converted",
        "offer_expiry", "offer_message_plain"
    ]

    df = df[[c for c in show_cols if c in df.columns]]

    # Show clean table
    st.dataframe(df, width="stretch", hide_index=True)

else:
    st.warning("⚠️ No messages found. Upload customers & generate offers!")

# -----------------------------
# Footer Analytics
# -----------------------------
st.divider()
st.subheader("📈 Campaign Insights")

if not df.empty:
    avg_propensity = df["propensity_score"].mean()
    avg_conversion = df["converted"].mean() * 100
    avg_click = df["clicked"].mean() * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Avg Propensity", f"{avg_propensity:.2f}")
    col2.metric("🟢 Click Rate", f"{avg_click:.1f}%")
    col3.metric("🎯 Conversion Rate", f"{avg_conversion:.1f}%")

    st.bar_chart(df.groupby("loyalty_status")[["clicked", "converted"]].mean())

