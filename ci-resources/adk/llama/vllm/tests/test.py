import sys
import requests
import json

def test_model_interaction(base_url):
    url = f'{base_url}/apps/weather_agent/users/1/sessions'
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
    expected_city = "Tokyo"
    data = {
        "appName": "weather_agent",
        "userId": "1",
        "sessionId": str(session_id),
        "newMessage": {
            "parts": [
                {
                    "text": f"What is the weather in {expected_city}?"
                }
            ],
            "role": "user"
        },
        "streaming": False
    }

    expected_functional_response = "Tokyo sees humid conditions with a high of 28 degrees Celsius (82 degrees Fahrenheit) and possible rainfall."

    response = requests.post(url, headers=headers, data=json.dumps(data))
    response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
    print("Status Code:", response.status_code)
    output = json.loads(response.text)

    actual_city = output[0]["content"]["parts"][0]["functionCall"]["args"]["city"]
    actual_functional_response = output[1]["content"]["parts"][0]["functionResponse"]["response"]["result"]

    assert expected_city == actual_city, f"Expected city != actual city. '{expected_city}' != '{actual_city}'."
    assert expected_functional_response == actual_functional_response, f"Expected functional response != actual functional response. '{expected_functional_response}' != '{actual_functional_response}'."
    print("PASS")


base_url = sys.argv[1]
test_model_interaction(base_url)
