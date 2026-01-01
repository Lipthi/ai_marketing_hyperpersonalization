# backend/rule_engine.py

import os 
import sys
import random
import json
from datetime import datetime, timezone, timedelta
from collections import Counter
from pathlib import Path
import joblib

# --- Hybrid Import ---
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from perk_manager import get_random_perk
    from message_generator import generate_message_with_llm
    from propensity_model import load_model, predict_batch, train_and_save_model
else:
    from .perk_manager import get_random_perk
    from .message_generator import generate_message_with_llm
    from .propensity_model import load_model, predict_batch, train_and_save_model

# Constants
LOYALTY_PRIORITY  = {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3}
PRODUCTS = ["shoes", "clothing", "haircare", "cosmetics", "accessories"]

## Simulation helper

def simulate_engagement(propensity_score: float, engagement_score: float):
    """
    Simulate click and conversion for generated offers.
    Retruns:
        clicked (0/1), converted (0/1)

    Explanation:
    - click probability increases with propensity_score and engagement_score
    - conversion only possible if clicked; conversion probability increases with propensity_score
    """

    # clamp inputs
    try:
        p = float(propensity_score)
    except Exception:
        p = 0.0
    try:
        e = float(engagement_score)
    except Exception:
        e = 0.0

    # Base probabilities
    base_click_prob = 0.05
    base_conversion_prob = 0.02

    # Combine signals (simple linear blend)
    click_prob = base_click_prob + (p * 0.25) + (e * 0.20)
    conversion_prob = base_conversion_prob + (p * 0.10)

    # Bound probabilities (0..0.9 to avoid extremes)
    click_prob = max(0.0, min(0.9, click_prob))
    conversion_prob = max(0.0, min(0.9, conversion_prob))

    clicked = int(random.random() < click_prob)

    # conversion only when clicked
    converted = int(random.random() < conversion_prob) if clicked else 0

    return clicked, converted


# --------------------------
# Helper: pick favourite product (most frequent)
# --------------------------
def _fav_product(customer):
    tx = customer.get("recent_transactions") or []
    if not tx:
        return "our products"
    products = [t.get("product", "our products") for t in tx]
    return max(set(products), key=products.count)

# --------------------------
# Helper: suggest upsell product
# --------------------------
def _suggest_upsell(customer):
    bought = set([t.get("product") for t in customer.get("recent_transactions", [])])
    options = list(set(PRODUCTS) - bought)
    return random.choice(options) if options else None

# --------------------------
# Helper: seasonal phrase
# --------------------------
def _seasonal_phrase(date=None):
    month = datetime.now().month
    if month in (11, 12):
        return "Holiday season - grab it before stocks run out!"
    if month in (2, 3):
        return "Spring special - fresh picks for you!"
    if month in (6, 7, 8):
        return "Summer sale - limited time offers!"
    return ""

# --------------------------
# Helper: compute base discount
# --------------------------
def _compute_discount_base(customer):
    engagement = customer.get("engagement_score", 0)
    churn = customer.get("churn", 0)
    loyalty = customer.get("loyalty_status", "Bronze")
    total_spent = customer.get("total_spent", 0)
    last_days = customer.get("last_purchase_days", 999)

    # --- Base discount logic by engagement/churn ---
    if churn == 1:
        base = 25  # higher base for churned users
    elif engagement >= 0.85:
        base = 12
    elif engagement >= 0.6:
        base = 10
    elif engagement >= 0.4:
        base = 8
    else:
        base = 10

    # --- Loyalty adjustments ---
    if loyalty in ("Gold", "Platinum"):
        # Loyal customers get small perks unless churning
        if churn == 1 or last_days > 90:
            base += 8  # retention boost
        else:
            base = min(base, 18)  # cap discount for healthy loyal customers
    elif loyalty == "Silver":
        base += 5
    elif loyalty == "Bronze":
        base += 8

    # --- Spending bonus ---
    if total_spent > 40000:
        base += 2
    elif total_spent > 20000:
        base += 1

    # --- Long inactivity boost ---
    if last_days > 90:
        base += 5

    # --- Add small random jitter ---
    jitter = random.choice([-2, -1, 0, 1, 2])
    discount = int(max(5, min(35, base + jitter)))  # cap at 35%

    return discount

# --------------------------
# Helper: expiry + urgency
# --------------------------
def _expiry_and_urgency(customer):
    now = datetime.now(timezone.utc)
    churn = customer.get("churn", 0)
    engagement = customer.get("engagement_score", 0)

    if churn == 1:
        expire_in_days = random.randint(2, 3)
        urgency = "Limited time — ends soon!"
    elif engagement >= 0.85:
        expire_in_days = random.randint(7, 14)
        urgency = "Exclusive early access — valid for a limited time."
    else:
        expire_in_days = random.randint(4, 10)
        urgency = "Hurry — offer expires soon."

    expiry_date = (now + timedelta(days=expire_in_days)).date().isoformat()
    return expiry_date, urgency

def _normalise_customer_input(c):
    """
    Ensure 'c' is a dict. If it's a JSON string, parse it.
    Provide safe defaults for required keys.
    """
    if isinstance(c, str):
        try:
            c = json.loads(c)
        except Exception:
            # fallback: wrap string into a minimal object so code doesn't break
            return {"customer_id": c, "name": str(c)}
    if not isinstance(c, dict):
        return {"customer_id": str(c), "name": str(c)}
    # Ensure required fields exist
    c.setdefault("customer_id", "unknown")
    c.setdefault("name", "Customer")
    c.setdefault("recent_transactions", [])
    c.setdefault("engagement_score", 0.0)
    c.setdefault("churn", 0)
    c.setdefault("loyalty_status", "Bronze")
    c.setdefault("propensity_score", 0.5)
    c.setdefault("total_spent", 0)
    c.setdefault("last_purchase_days", 999)
    # phone may be present — keep as is or empty string
    if "phone" not in c:
        c["phone"] = ""
    return c

# --------------------------
# Offer builder
# --------------------------

def build_offer(customer, use_ml=False, model=None):

    customer = _normalise_customer_input(customer)

    customer_id = customer["customer_id"]
    name = customer.get("name", "Customer").split()[0]
    email = customer.get("email")
    phone = customer.get("phone") or ""
    fav_product = _fav_product(customer)
    upsell_product = _suggest_upsell(customer)
    base_discount = _compute_discount_base(customer)
    expiry_date, urgency = _expiry_and_urgency(customer)
    seasonal = _seasonal_phrase()
    loyalty = customer.get("loyalty_status", "Bronze")
    churn_flag = int(customer.get("churn", 0))
    engagement = float(customer.get("engagement_score", 0))
    propensity = float(customer.get("propensity", customer.get("propensity_score", 0.5)))

    discount = base_discount
    # Adjust discount slightly based on conversion likelihood
    if propensity > 0.8:
        discount = max(5, discount - 3)  # less discount if very likely to buy anyway
    elif propensity < 0.4:
        discount = min(35, discount + 5)  # sweeten deal for low-probability customers

    offer_code = f"{customer_id[-4:]}-{random.randint(100, 999)}"

    # Tracking URLs
    track_click_url = f"http://127.0.0.1:8000/track/click/{customer_id}/{offer_code}"
    redeem_url = f"http://127.0.0.1:8000/track/convert/{customer_id}/{offer_code}"


    # subject & message
    if churn_flag == 1:
        subject = f"{name}, we miss you — {discount}% off just for you"
        main = f"Hi {name}, enjoy {discount}% OFF on {fav_product}! Use code {offer_code} at checkout by {expiry_date}."
    elif engagement >= 0.85 and LOYALTY_PRIORITY.get(loyalty, 0) >= 2:
        subject = f"Exclusive gift for {loyalty} members, {name}"
        main = f"Hi {name}, as a {loyalty} member, you get a free gift with your next {fav_product} purchase! Code: {offer_code}, valid until {expiry_date}."
    else:
        subject = f"{discount}% OFF on {fav_product} — limited time"
        main = f"Hi {name}, enjoy {discount}% OFF on {fav_product}! Code: {offer_code}, valid until {expiry_date}."

    extras = [urgency] if urgency else []
    if seasonal:
        extras.append(seasonal)
    if upsell_product:
        extras.append(f"P.S. Check out our {upsell_product} collection!")

    #adding perks
    perk_message = ""
    if loyalty in ("Gold", "Platinum"):
        perk_message = get_random_perk(loyalty, fav_product)
    if perk_message:
        extras.append(perk_message)

    raw_message = " ".join([main] + extras)

    cta_block = (
        f"\n\n👉 View offer: {track_click_url}"
        f"\n🎁 Redeem here: {redeem_url}"
    )

    final_message = raw_message + cta_block


    #Offer dictionary
    offer = {
        "offer_id": offer_code,
        "customer_id": customer_id,
        "name": name,
        "email": email,
        "phone": phone,
        "discount_pct": discount,
        "offer_type": "discount",
        "expiry_date": expiry_date,
        "urgency": urgency, 
        "seasonal_phrase": seasonal,
        "subject": subject,
        "offer_message": final_message,
        "loyalty_status": loyalty,
        "engagement_score": engagement,
        "churn": churn_flag,
        "propensity_score": propensity,
        "track_click_url": track_click_url,
        "redeem_url": redeem_url
    }

# --- WhatsApp-safe version (URL encoded) ---
    from urllib.parse import quote
    # Clean version for display (Streamlit)
    offer["offer_message_plain"] = final_message
# WhatsApp-safe encoded version (for sending links)
    offer["offer_message_whatsapp"] = quote(final_message)

# Simulating engagement here (single source of truth)
    clicked, converted = simulate_engagement(propensity, engagement)
    offer["clicked"] = clicked
    offer["converted"] = converted

    return offer

# --------------------------
# Bulk offers
# --------------------------
def generate_bulk_offers(customers, use_ml=False):
    """
    Generates personalized offers for all customers.
    Includes a propensity score (likelihood to convert) for each customer.
    """

    normalized_customers = []
    for c in customers:
        normalized_customers.append(_normalise_customer_input(c))

    # ---  Load or train the propensity model ---
    model = load_model()
    if model is None:
        print("⚙️ No saved model found — training a new one...")
        model = train_and_save_model()

    # ---  Predict propensity for all customers ---
    try:
        probs = predict_batch(customers, model=model)
        for c, p in zip(customers, probs):
            c["propensity"] = float(p)
            # also keep legacy key
            c["propensity_score"] = float(p)
    except Exception as e:
        print(f"⚠️ Propensity prediction failed: {e}")
        for c in customers:
            c["propensity"] = 0.5  # default neutral score
            c["propensity_score"] = 0.5
    offers = []

    for cust in normalized_customers:
        try:
            offer = build_offer(cust, use_ml=use_ml)
            offers.append(offer)
        except Exception as e:
            cid = cust.get("customer_id", str(cust))
            print(f"⚠️ Failed to build offer for {cid}: {e}")
            continue

    return offers

