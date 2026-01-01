import os, sys
import json
import random
from pathlib import Path

# --- Hybrid Import ---
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Path to perk templates JSON file
PERK_PATH = Path(__file__).resolve().parent.parent / "data" / "perk_templates.json"

def get_random_perk(loyalty_tier: str, product_category: str):
    """
    Loads a random perk message from JSON and fills placeholders dynamically.
    Only applies for Gold and Platinum customers.
    Returns a string like:
      'As a valued Gold member, enjoy early access to our collection!'
    """
    try:
        with open(PERK_PATH, "r", encoding="utf-8") as f:
            perk_templates = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ Perk template file not found at {PERK_PATH}. Using fallback perks.")
        perk_templates = {
            "VIP Preview": [
                "As a valued {loyalty_tier} member, enjoy a VIP preview of our {product_category} range!"
            ],
            "Exclusive Gift": [
                "You’ve unlocked an exclusive gift as part of your {loyalty_tier} membership in {product_category}!"
            ],
            "Early Access": [
                "As a {loyalty_tier} member, you get early access to our new {product_category} collection!"
            ],
        }
    except Exception as e:
        print(f"⚠️ Failed to load perk templates: {e}")
        return ""

    # Only Gold and Platinum get perks
    if loyalty_tier not in ("Gold", "Platinum"):
        return ""

    # Choose a perk type randomly
    possible_perks = ["VIP Preview", "Exclusive Gift", "Early Access"]
    perk_type = random.choice(possible_perks)

    # Get templates for that perk type
    templates = perk_templates.get(perk_type, [])
    if not templates:
        return ""

    # Pick one template and fill placeholders
    template = random.choice(templates)
    message = template.format(
        loyalty_tier=loyalty_tier,
        product_category=product_category or "our collection"
    )

    return message
