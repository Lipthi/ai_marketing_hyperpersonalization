# whatsapp_sender.py
import os
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
FROM_WHATSAPP = os.getenv("TWILIO_WHATSAPP_FROM")

client = Client(ACCOUNT_SID, AUTH_TOKEN)

def send_whatsapp_message(phone, message):
    try:
        msg = client.messages.create(
            from_=FROM_WHATSAPP,
            body=message,
            to=f"whatsapp:{phone}"
        )
        return {"status": "sent", "message_sid": msg.sid}
    except Exception as e:
        return {"status": "error", "error": str(e)}
