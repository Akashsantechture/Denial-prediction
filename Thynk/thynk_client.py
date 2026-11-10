
import os
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://stage-thynk2-api.santechture.com/thynk-gateway"

LOGIN_URL = f"{BASE_URL}/auth-service/auth/login-by-key"

VALIDATE_URL = (
    f"{BASE_URL}/integration-service/api/v1/services/validate"
)

def login():
    payload = {
        "username": os.getenv("THYNK_USERNAME"),
        "companyKey": os.getenv("THYNK_COMPANY_KEY")
    }

    response = requests.post(
        LOGIN_URL,
        json=payload,
        timeout=30
    )

    print("Login status:", response.status_code)
    response.raise_for_status()

    result = response.json()
    print("Login response:", result)

    # Check the response structure before extracting token.
    print("Login response keys:", list(result.keys()))

    token = result["data"]["auth"]["token"]
    print("Token received:", token)

    print("Authentication successful!")
    return token


# import json
# from pathlib import Path

# def validate_claim(token, claim_payload):

#     params = {
#         "name": "KSA_TEST_CLAIM",
#         "type": "CHI",
#         "handleType": "API",
#         "validationType": "SUBMISSION",
#         "resubmission": "false",
#         "actions": "false",
#         "showDenialCode": "true",
#         "ShowOnlyRulesWithDenialCode": "false"
#     }

#     headers = {
#         "Authorization": f"Bearer {token}",
#         "Content-Type": "application/json"
#     }

#     response = requests.post(
#         VALIDATE_URL,
#         params=params,
#         headers=headers,
#         json=claim_payload,
#         timeout=120
#     )

#     print("Validate status:", response.status_code)

#     try:
#         result = response.json()
#     except ValueError:
#         print("Non-JSON response:", response.text[:2000])
#         response.raise_for_status()
#         return None

#     if not response.ok:
#         print(json.dumps(result, indent=4))
#         response.raise_for_status()

#     print("\nTHYNK VALIDATION RESPONSE")
#     print(json.dumps(result, indent=4))

#     return result



if __name__ == "__main__":
    token = login()
    print("Token received:", bool(token))
    # with open("Thynk/test_claim.json", "r") as f:
    #     claim_payload = json.load(f)

    # result = validate_claim(token, claim_payload)

    # print("\nClaim validation completed!")
