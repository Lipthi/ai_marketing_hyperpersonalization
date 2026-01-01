# data_generator.py
import random
import json
import pandas as pd
from faker import Faker
import os

fake = Faker()

PRODUCTS = ["shoes", "clothing", "haircare", "cosmetics", "accessories"]
LOYALTY_TIERS = ["Bronze", "Silver", "Gold", "Platinum"]

LOYALTY_WEIGHTS = {
    "Bronze": 0.1,
    "Silver": 0.3,
    "Gold": 0.6,
    "Platinum": 0.9
}

def generate_email_from_name(name):
    username = name.lower().replace(" ", "")   # remove spaces, lowercase
    number = random.randint(10, 99)            # 2-digit random number
    return f"{username}{number}@example.com"

def generate_phone_number():
    """Generate WhatsApp-friendly Indian number."""
    return "+91" + str(random.randint(6000000000, 9999999999))

def generate_customer(customer_id):
    """Generate one customer with transactions."""
    num_transactions = random.randint(1, 6)

    # Generate transactions
    transactions = [
        {"product": random.choice(PRODUCTS), "amount": round(random.uniform(200, 5000), 2)}
        for _ in range(num_transactions)
    ]

    total_spent = sum(t["amount"] for t in transactions)
    avg_order_value = round(total_spent / num_transactions, 2)
    last_purchase_days = random.randint(1, 120)
    loyalty_status = random.choice(LOYALTY_TIERS)

    #Engagement score calculation (0 - 1 scale)
    enagagement_score = (
        (1 - (last_purchase_days / 120)) * 0.4 +
        min(total_spent / 50000, 1) * 0.3 +
        (LOYALTY_WEIGHTS[loyalty_status]) * 0.2 +
        min(num_transactions / 10, 1) * 0.1
    )
    engagement_score  = round(min(1, enagagement_score), 2)

    churn = 1 if (last_purchase_days > 60 and enagagement_score < 0.4) else 0

    name = fake.name()
    email = generate_email_from_name(name)
    
    customer = {
        "customer_id": f"c{customer_id:06d}",
        "name": name,
        "email": email,
        "phone": generate_phone_number(),
        "age": random.randint(18, 70),
        "gender": random.choice(["Male", "Female"]),
        "location": fake.city(),
        "income": random.choice(["Low", "Medium", "High"]),
        "total_spent": round(total_spent, 2),
        "avg_order_value": avg_order_value,
        "loyalty_status": loyalty_status,
        "last_purchase_days": last_purchase_days,
        "channel_preference": "Email",
        "recent_transactions": transactions, # keep as JSON array
        "engagement_score": engagement_score,
        "churn": churn
    }
    return customer

def generate_customers(n=10, folder="../data/processed"):
    os.makedirs(folder, exist_ok=True)

    # Generate all customers
    data = [generate_customer(i) for i in range(1, n+1)]

    # Save JSON array
    json_path = f"{folder}/customers.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    # Save CSV
    df = pd.DataFrame(data)
    csv_path = f"{folder}/customers.csv"
    df.to_csv(csv_path, index=False)

    print(f"Generated {n} customers")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    return df

if __name__ == "__main__":
    generate_customers(10)
