# generate_messages.py
import json
import os
import sys
import time

# --- Hybrid Import ---
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from rule_engine import generate_bulk_offers
    from message_generator import generate_message_with_llm
else:
    from .rule_engine import generate_bulk_offers
    from .message_generator import generate_message_with_llm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOMER_PATH = os.path.join(BASE_DIR, "../data/processed/customers.json")
MESSAGE_PATH = os.path.join(BASE_DIR, "../data/processed/messages.json")
PROGRESS_PATH = os.path.join(BASE_DIR, "../logs/progress.json")

def write_progress(done, total, name=""):
    progress = {
        "completed": done,
        "total": total,
        "last_customer": name,
        "timestamp": time.time()
    }
    os.makedirs(os.path.dirname(PROGRESS_PATH), exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump(progress, f)

def append_message(record):
    """Append each generated message"""
    if os.path.exists(MESSAGE_PATH):
        with open(MESSAGE_PATH, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(record)
    with open(MESSAGE_PATH, "w") as f:
        json.dump(data, f, indent=2)


def main():
    # 1. Load customers
    with open(CUSTOMER_PATH, "r") as f:
        customers = json.load(f)

    total = len(customers)
    write_progress(0, total, "Starting...")

    # 2. Reset messages
    os.makedirs(os.path.dirname(MESSAGE_PATH), exist_ok=True)
    with open(MESSAGE_PATH, "w") as f:
        json.dump([], f)

    # 3. Generate ALL structured offers
    all_offers = []

    for i, cust in enumerate(customers, 1):
        offers = generate_bulk_offers([cust])
        if not offers:
            continue
        all_offers.append(offers[0])

    if not all_offers:
        print("⚠️ No offers generated.")
        return

    # 4. Enhance messages with LLM and save
    for i, offer in enumerate(all_offers, 1):
        name = offer["name"]
        loyalty = offer.get("loyalty_status", "Bronze")
        fav_product = offer.get("offer_message", "").split(
            "on ")[-1].split("!")[0].strip()
        seasonal = offer.get("seasonal_phrase", "")

        print(f"Processing customer {i}/{total}: {name} ({loyalty})", flush=True)

        try:
            enhanced_msg = generate_message_with_llm(
                raw_message=offer["offer_message"],
                customer_name=name,
                loyalty_status=loyalty,
                fav_product=fav_product,
                seasonal_phrase=seasonal
            )
        except KeyboardInterrupt:
            print("\n⏹️ Process interrupted by user.")
            sys.exit(0)
        except Exception as e:
            print(f"\n⚠️ Error for {name}: {e}")
            enhanced_msg = offer["offer_message"]

        offer["final_message"] = enhanced_msg.strip() + "\n\n" + (
            f"👉 View offer: {offer['track_click_url']}\n"
            f"🎁 Redeem here: {offer['redeem_url']}"
        )

        offer["offer_message_plain"] = enhanced_msg.strip()
        offer["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        append_message(offer)
        write_progress(i, total, name)

        print("✅")
        print(f"➡️ Enhanced message for {name} ({loyalty}): {enhanced_msg}\n")

    print("🎉 All messages generated.")


# -----------------------------
# STREAMLIT-SAFE GENERATOR
# -----------------------------

def run_generation_stream(customers, on_offer_callback):
    """
    Generates offers one-by-one and calls the callback after each offer.
    Safe to use inside Streamlit (NO subprocess).
    """

    total = len(customers)

    # Reset messages.json ONCE
    os.makedirs(os.path.dirname(MESSAGE_PATH), exist_ok=True)
    with open(MESSAGE_PATH, "w") as f:
        json.dump([], f)

    write_progress(0, total, "Starting...")

    for i, cust in enumerate(customers, start=1):
        offers = generate_bulk_offers([cust])
        if not offers:
            continue

        offer = offers[0]
        name = offer.get("name", "Customer")

        try:
            enhanced_msg = generate_message_with_llm(
                raw_message=offer["offer_message"],
                customer_name=name,
                loyalty_status=offer.get("loyalty_status", "Bronze"),
                fav_product="",
                seasonal_phrase=offer.get("seasonal_phrase", "")
            )
        except Exception:
            enhanced_msg = offer["offer_message"]

        offer["final_message"] = enhanced_msg.strip() + "\n\n" + (
            f"👉 View offer: {offer['track_click_url']}\n"
            f"🎁 Redeem here: {offer['redeem_url']}"
        )

        offer["offer_message_plain"] = enhanced_msg.strip()
        offer["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        append_message(offer)
        write_progress(i, total, name)

        on_offer_callback(offer, i, total)

if __name__ == "__main__":
    main()
