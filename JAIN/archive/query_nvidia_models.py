import requests
import json
import os

API_KEY = "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"
URL = "https://integrate.api.nvidia.com/v1/models"

def main():
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json"
    }
    
    response = requests.get(URL, headers=headers)
    
    os.makedirs("JAIN/artifacts", exist_ok=True)
    out_path = "JAIN/artifacts/nvidia_available_models.json"
    
    if response.status_code == 200:
        data = response.json()
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Successfully saved {len(data.get('data', []))} models to {out_path}")
        
        # Look for deepseek models
        deepseek_models = [m['id'] for m in data.get('data', []) if 'deepseek' in m['id'].lower()]
        print("DeepSeek models found:", deepseek_models)
    else:
        print(f"Failed to query models. Status: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    main()
