import requests
import json

url = "http://127.0.0.1:5000/chat"
headers = {"Content-Type": "application/json"}
data = {"message": "Apa visi Program Studi Teknologi Sains Data?"}

try:
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        print("Response from Chatbot:")
        print(response.json())
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Failed to connect: {e}")
