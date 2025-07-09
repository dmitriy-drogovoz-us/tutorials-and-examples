curl http://127.0.0.1:8000/v1/chat/completions \
-X POST \
-H "Content-Type: application/json" \
-d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
        {
        "role": "system",
        "content": "You are a helpful agent that provides weather report in a city using a tool.\nThe user will provide a city name in a JSON format like {\"city\": \"city_name\"}.\n1. Extract the city name.\n2. Use the `get_weather` tool to find the weather.\n3. Answer on user request based on the weather"
        },
        {
        "role": "user",
        "content": "I am going to vacation in Ottawa. Should I get a jucket?"
        }
    ],
    "tools": [
    {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Retrieves the weather condition of a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string"
                }
            }
        }
    }
    }
]
}'
