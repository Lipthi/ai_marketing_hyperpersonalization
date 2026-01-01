# backend/main.py
import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from whatsapp_sender import send_whatsapp_message

from generate_messages import main as generate_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MESSAGE_PATH = os.path.join(BASE_DIR, "../data/processed/messages.json")

app = FastAPI(title="Hyperpersonalisation API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_messages():
    if not os.path.exists(MESSAGE_PATH):
        return []
    with open(MESSAGE_PATH, "r") as f:
        return json.load(f)

def save_messages(data):
    with open(MESSAGE_PATH, "w") as f:
        json.dump(data, f, indent=2)

@app.get("/")
def home():
    return {"message": "Hyperpersonalisation API is running!"}

# --------------------------
# GENERATE MESSAGES
# --------------------------
@app.post("/generate-offers")
async def generate_offers():
    try:
        generate_pipeline()
        return {"message": "Messages generated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------
# PREVIEW
# --------------------------
@app.get("/preview")
async def preview():
    data = load_messages()
    if not data:
        raise HTTPException(status_code=404, detail="No data available")

    preview_cols = [
        "customer_id", "name", "loyalty_status", "discount_pct",
        "engagement_score", "propensity_score", "clicked", "converted"
    ]

    cleaned = [{k: d.get(k) for k in preview_cols} for d in data]
    return cleaned

# --------------------------
# OFFERS (INCLUDES CTA URLs)
# --------------------------
@app.get("/offers")
async def offers():
    data = load_messages()
    if not data:
        raise HTTPException(status_code=404, detail="No messages found")

    for d in data:
        cid = d.get("customer_id")
        oid = d.get("offer_id")

        d["track_url"] = f"http://127.0.0.1:8000/track/click/{cid}/{oid}"
        d["redeem_url"] = f"http://127.0.0.1:8000/track/convert/{cid}/{oid}"

    final_cols = [
        "customer_id", "name", "discount_pct", "loyalty_status",
        "clicked", "converted", "propensity_score", "engagement_score",
        "final_message", "generated_at", "track_url", "redeem_url"
    ]

    filtered = [{k: d.get(k) for k in final_cols} for d in data]
    return filtered

# ----------------------------------------------------
# TRACK CLICK  (customer_id + offer_id combination)
# ----------------------------------------------------
@app.get("/track/click/{customer_id}/{offer_id}")
async def track_click(customer_id: str, offer_id: str):

    data = load_messages()
    updated = False

    for msg in data:
        if msg.get("customer_id") == customer_id and msg.get("offer_id") == offer_id:
            msg["clicked"] = 1
            updated = True
            break

    if not updated:
        raise HTTPException(status_code=404, detail="Offer not found")

    save_messages(data)

    return HTMLResponse(f"""
        <html>
        <body style='font-family: Arial; padding:20px;'>
            <h2>Offer Click Recorded!</h2>
            <p>Thanks for viewing this offer.</p>
            <a href="/track/convert/{customer_id}/{offer_id}">
                <button style="padding:10px 20px;">Redeem Now</button>
            </a>
        </body>
        </html>
    """)

# ----------------------------------------------------
# TRACK CONVERSION (customer_id + offer_id)
# ----------------------------------------------------
@app.get("/track/convert/{customer_id}/{offer_id}")
async def track_convert(customer_id: str, offer_id: str):

    data = load_messages()
    updated = False

    for msg in data:
        if msg.get("customer_id") == customer_id and msg.get("offer_id") == offer_id:
            msg["converted"] = 1
            updated = True
            break

    if not updated:
        raise HTTPException(status_code=404, detail="Offer not found")

    save_messages(data)

    return HTMLResponse("""
        <html>
        <body style='font-family: Arial; padding:20px;'>
            <h2>Thank You ❤️</h2>
            <p>Your redemption has been recorded successfully.</p>
        </body>
        </html>
    """)

# ----------------------------------------------------
# SEND WHATSAPP
# ----------------------------------------------------
@app.get("/send-whatsapp/{customer_id}")
def send_whatsapp(customer_id: str):

    data = load_messages()
    row = next((d for d in data if d["customer_id"] == customer_id), None)

    if not row:
        raise HTTPException(status_code=404, detail="Customer not found")

    phone = row.get("phone")
    message = row.get("final_message")

    if not phone:
        raise HTTPException(status_code=400, detail="Phone number missing")

    result = send_whatsapp_message(phone, message)

    return {
        "sent_to": phone,
        "customer_id": customer_id,
        "whatsapp_result": result
    }
