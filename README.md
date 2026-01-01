# Hyperpersonalization Marketing System

This project demonstrates a **hyperpersonalization marketing system** that predicts how likely each customer is to respond to an offer and delivers tailored messages according to customer behavior data and an ML propensity model.

## The system combines:

- Rule-based campaign logic  
- Machine Learning–driven propensity scoring  
- LLM-powered message enhancement  
- Real-time UI monitoring  
- WhatsApp campaign delivery (Twilio)  

## Synthetic customer data with:

- Loyalty tiers (Bronze → Platinum)  
- Engagement score  
- Churn flag  
- Transaction history  
- Spending behavior  

## Deterministic business rules generate base offers:

- Discount % based on loyalty & churn risk  
- Product category alignment  
- Seasonal phrasing  
- Ensures explainability and control before AI enhancement  

## LLM-Powered Message Generation

- Base offers are enhanced using an LLM (via Ollama)  
- Human-like tone  
- Personalization using customer context  
- Fallback logic ensures system reliability if LLM fails  

## Propensity Scoring (Machine Learning)

- Logistic Regression–based propensity model predicts:  
  *“How likely is this customer to convert if shown an offer?”*  
- Uses engineered features such as engagement score, churn, transaction count, recency & spend  

## Offline Model Retraining (Real-World Design)

- Uses accumulated customer data  
- Avoids unstable live updates  
- Offline learning improves future predictions  

## Real-Time Dashboard (Streamlit)

- Live offer generation per customer  
- Filters by loyalty tier, engagement score, churn / active status  

## WhatsApp Campaign Delivery

- Integrated with Twilio WhatsApp Sandbox  
- Send personalized offers directly to customers  

## Tracking clicks and redeeming

- Each individual offer consists of a URL  
- Clicking updates the UI that the customer has clicked on the offer, which is suitable for future training  

## Setup & Run Instructions

1. **Clone the repository**  
2. **Create & activate a virtual environment**  
3. **Install dependencies**  
pip install -r requirements.txt
4. **Run tha backend scripts**
# Generate synthetic customers
python backend/data_generator.py
5. **Run the FastAPI endpoints**
uvicorn main:app --reload
6. **Run the dashboard**
cd dashboard
streamlit run app.py
7. **Navigate the UI**
Upload the customers.json file
Explore customer characteristics and analytics
Generate offers and send them via WhatsApp messaging




