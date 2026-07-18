import google.generativeai as genai

genai.configure(api_key="Your Api Key")

print("Tere liye ye models available hain:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
