# test_gemini.py

from google import genai

# Replace with your API key or use an environment variable
API_KEY = "api key"



client = genai.Client(api_key=API_KEY)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What model are you?"
)

print(response.text)

# try:
#     client = genai.Client(api_key=API_KEY)

#     response = client.models.generate_content(
#         model="gemini-2.5-flash",
#         contents="Reply with exactly: Gemini API is working"
#     )

#     print("\n✅ API Connection Successful")
#     print("Response:")
#     print(response.text)

# except Exception as e:
#     print("\n❌ API Connection Failed")
#     print(f"Error: {e}")