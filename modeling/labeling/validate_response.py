import json
import re

def clean_json_response(response_text):
    # Sometimes LLMs wrap in ```json ... ```
    if "```json" in response_text:
        match = re.search(r"```json(.*?)```", response_text, re.DOTALL)
        if match:
            response_text = match.group(1).strip()
    elif "```" in response_text:
        match = re.search(r"```(.*?)```", response_text, re.DOTALL)
        if match:
            response_text = match.group(1).strip()
    return response_text

def validate_response(response_text, expected_count):
    cleaned = clean_json_response(response_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return False, f"JSON parsing failed: {e}", []
        
    if not isinstance(data, list):
        return False, "Output is not a JSON array", []
        
    if len(data) != expected_count:
        return False, f"Expected {expected_count} objects, got {len(data)}", []
        
    required_keys = {
        "technical_fit", "product_fit", "behavioral_fit", 
        "career_fit", "fit_score", "fit_label", 
        "honeypot_probability", "honeypot_label", "confidence", "reasoning"
    }
    
    for idx, obj in enumerate(data):
        missing = required_keys - set(obj.keys())
        if missing:
            return False, f"Object at index {idx} missing keys: {missing}", []
            
    return True, "Success", data
