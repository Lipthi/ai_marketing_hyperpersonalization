This project demonstrates a hyperpersonalization markerting system that predicts how likely each customer is to respond to an offer and delivers tailored messages acccording to customer behavior data and an ML proensity model

The system combines:
-> Rule-based campaign logic
-> Machine Learning–driven propensity scoring
-> LLM-powered message enhancement
-> Real-time UI monitoring
-> WhatsApp campaign delivery (Twilio)

--> Synthetic customer data with:
Loyalty tiers (Bronze → Platinum), Engagement score, Churn flag, Transaction history, Spending behavior

--> Deterministic business rules generate base offers:
Discount % based on loyalty & churn risk, Product category alignment, Seasonal phrasing, Ensures explainability and control before AI enhancement.

--> LLM-Powered Message Generation
Base offers are enhanced using an LLM (via Ollama), Human-like tone, Personalization using customer context, Fallback logic ensures system reliability if LLM fails.

--> Propensity Scoring (Machine Learning)
Logistic Regression–based propensity model predicts:
“How likely is this customer to convert if shown an offer?”
Uses engineered features such as: Engagement score, Churn, Transaction count, Recency & spend

--> Offline Model Retraining (Real-World Design)
Uses accumulated customer data, Avoids unstable live updates, Offline learning improves future predictions

--> Real-Time Dashboard (Streamlit)
Live offer generation per customer, Filters by: Loyalty tier, Engagement score, Churn / Active status

--> WhatsApp Campaign Delivery
Integrated with Twilio WhatsApp Sandbox, Send personalized offers directly to customers

-->Tracking clicks and redeeming
Each individual offer sonsists of a URL, clicking on which updtaes the UI that the customer has clicked on the offer which is suitable for future training.

Setup & Run Instructions
-> Clone the repository
-> Create & activate a virtual environment
-> Install dependencies
pip install -r requirements.txt
-> Run the backend scripts
Generate synthetic customers:
python backend/data_generator.py
-> Run the FastApi endpoints
uvicorn main:app --reload
-> Run the dashboard
cd dashboard
streamlit run app.py
-> Navigate the UI and upload the customers.json file and have a look at the customer characteristics and analytics, generate offers and send them via whatsapp messaging.
