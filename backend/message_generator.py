# backend/message_generator.py

import os, sys
import requests
import textwrap
import json
import random

# --- Hybrid Import ---
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
TIMEOUT = 240

# Tone mapping based on loyalty
LOYALTY_TONES = {
    "Bronze": "friendly and motivating",
    "Silver": "warm and approachable",
    "Gold": "refined and premium",
    "Platinum": "exclusive and elegant"
}

CTA_VARIANTS = [
    "Shop now!",
    "Grab yours today!",
    "Don't miss out!",
    "Hurry, limited time!",
    "Check it out!"
]

EMOJIS = ["✨", "🎉", "💼", "🛍️", "🌟", "💖", "🔥", "💎"]


def generate_message_with_llm(
    raw_message: str,
    customer_name: str,
    loyalty_status: str = "Bronze",
    fav_product: str = None,
    seasonal_phrase: str = "",
    perk_message: str = ""
) -> str:
    """
    Sends raw offer message to Ollama (Mistral model) and returns
    a concise, personalized, and tone-adjusted version.
    """

    tone = LOYALTY_TONES.get(loyalty_status, "friendly")
    cta = random.choice(CTA_VARIANTS)
    emoji_block = " ".join(random.sample(EMOJIS, 2))

    if loyalty_status in ("Gold", "Platinum"):
        emoji_block = random.choice(["✨", "💎", "🌟"])

    prompt = textwrap.dedent(f"""
    You are a world-class AI marketing assistant.

    Rewrite the promotional message below in a **{tone} tone**.
    Address the customer by their first name: {customer_name}.
    Highlight their loyalty tier (“{loyalty_status} Member”) gracefully in the first line if possible.
    If favorite product (“{fav_product}”) is mentioned, make it sound personal.
    Optionally include this seasonal phrase naturally: "{seasonal_phrase}"
    {f'Optionally include this perk naturally: "{perk_message}"' if perk_message else ""}
    Keep the final message **under 60 words**, engaging, clear, and aligned with the tone.
    Add emojis subtly: {emoji_block}.
    End with a persuasive call-to-action: {cta}.
    Never invent new offers or discounts.

    Original Message:
    {raw_message}
    """)

    payload = {
    "model": "mistral:latest",
    "prompt": prompt,
    "stream": False,  
    "temperature": 0.7
}

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=TIMEOUT) as response:
            response.raise_for_status()

            data = response.json()
            generated_text = data.get("response", "").strip()

            if not generated_text:
                print("⚠️ Empty LLM response — using fallback message.")
                return raw_message

            #print(f" Generated message for {customer_name}: {generated_text}")
            return generated_text

    except Exception as e:
        print(f"⚠️ Ollama failed: {e} — using fallback message.")
        return raw_message

        
