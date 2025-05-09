import twilio
import requests
from twilio.rest import Client
import time
from datetime import datetime
import os

# Twilio credentials
sid = 'YOUR_TWILIO_SSID'
token = 'YOUR_TWILIO_API_KEY'
twilio_number = 'PURCHASED_NUMBER'
number = 'VERIFIED_NUMBER'
client = Client(sid, token)

#for image uploads and getting links to image...............................................................................
import requests

def upload(image_path):
    try:
        with open(image_path, 'rb') as img:
            response = requests.post('https://store1.gofile.io/uploadFile', files={'file': img})
            if not response.content:
                print("No response content from GoFile.")
                return None

            data = response.json()
            if "data" in data and "downloadPage" in data["data"]:
                return data["data"]["downloadPage"]
            else:
                print("Unexpected response structure:", data)
                return None

    except Exception as e:
        print("Upload error:", e)
        return None

#For sending message and link to phone vis SMS....................................................................................

def send_message(image_url):
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message_body = f"[VISIONLOCK ALERT]\nTime: {timestamp}\n An unrecognized individual was detected near your secured premises.\n click the link below to view captured image\n  View Image: {image_url}"

    message = client.messages.create(
        body = message_body,
        from_= twilio_number,
        to= number    
    )
    print("Message SID:", message.sid)
