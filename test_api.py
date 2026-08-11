import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_FOOTBALL_KEY")

url = "https://v3.football.api-sports.io/standings"
headers = {"x-apisports-key": API_KEY}
params = {"league": 1, "season": 2026}

response = requests.get(url, headers=headers, params=params)
data = response.json()

# Check for errors first
print("Status:", response.status_code)
print("Errors:", data.get("errors"))
print("Results count:", data.get("results"))
print()

# Print the raw structure so we can see what we're working with
import json
print(json.dumps(data, indent=2)[:3000])  # first 3000 chars