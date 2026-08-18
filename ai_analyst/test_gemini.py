import os

from dotenv import load_dotenv
from google import genai
from google.auth import default


load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv(
    "GOOGLE_CLOUD_LOCATION",
    "us-central1"
)
MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

print("========================================")
print("Google Gemini Authentication Test")
print("========================================")

print(f"Project : {PROJECT_ID}")
print(f"Location: {LOCATION}")
print(f"Model   : {MODEL}")

# ------------------------------------------------------------
# Load Application Default Credentials
# ------------------------------------------------------------

credentials, detected_project = default(
    scopes=[
        "https://www.googleapis.com/auth/cloud-platform"
    ]
)

print(f"Detected project: {detected_project}")
print("Google authentication loaded successfully.")

# ------------------------------------------------------------
# Create Vertex AI client
# ------------------------------------------------------------

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID or detected_project,
    location=LOCATION,
    credentials=credentials,
)

# ------------------------------------------------------------
# Test Gemini
# ------------------------------------------------------------

print("\nCalling Gemini...")

response = client.models.generate_content(
    model=MODEL,
    contents="Explain SHAP in one simple sentence.",
)

print("\n========================================")
print("GEMINI RESPONSE")
print("========================================")
print(response.text)
print("========================================")