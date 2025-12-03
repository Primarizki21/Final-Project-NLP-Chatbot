
import google.generativeai as genai
import os

# Gunakan API Key yang sudah Anda set sebelumnya
api_key = "AIzaSyDZhsoRSqh6x4Fy-tO8m9ML6HkXMc4K40c"
genai.configure(api_key=api_key)

print("Checking available models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- Found model: {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")

