import sys
import requests
import json

def test_model_interaction(base_url):
    app_name = "example-agent"
    user_id = "user1"
    url = f'{base_url}/apps/{app_name}/users/{user_id}/sessions'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }

    session_id = None
    response = requests.post(url, headers=headers)
    response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
    session_id = response.json()["id"]
    print("Status Code:", response.status_code)
    print("Response Body:", response.json())


    assert session_id is not None, "The Session ID is None."

    url = f'{base_url}/run'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {
        "app_name": app_name,
        "user_id": user_id,
        "session_id": str(session_id),
        "new_message": {
            "parts": [
                {
                    "text": "What is the weather like in Seattle?"
                }
            ],
            "role": "user"
        },
        "streaming": False
    }

    expected_city  = "Seattle"
    expected_functional_call_result = "The weather in Seattle is currently 12°C with rainy conditions."

    response = requests.post(url, headers=headers, data=json.dumps(data))
    response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
    print("Status Code:", response.status_code)
    messages = response.json()
    actual_city = messages[0]["content"]["parts"][0]["functionCall"]["args"]["city"]
    actual_functional_call_result = messages[1]["content"]["parts"][0]["functionResponse"]["response"]["result"]
    assert actual_city == expected_city, f"Error. The actual city({actual_city}) != the expected ({expected_city}) one."
    assert actual_functional_call_result == expected_functional_call_result, f"Error. The actual functional response({actual_functional_call_result}) != the expected ({expected_functional_call_result}) one."
    print(f"The actual text response: {messages}")
    print("PASS")


base_url = sys.argv[1]
test_model_interaction(base_url)
