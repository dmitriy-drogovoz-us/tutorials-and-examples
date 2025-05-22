# Full example code for the basic weather agent
# --- Full example code demonstrating LlmAgent with Tools ---
import asyncio
import os
import json # Needed for pretty printing dicts

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from litellm.rerank_api.main import httpx
from pydantic import BaseModel, Field


# --- 1. Define Constants ---
APP_NAME = "agent_comparison_app"
USER_ID = "test_user_456"
SESSION_ID_TOOL_AGENT = "session_tool_agent_xyz"
MODEL_NAME = "Llama-3.1-8B-Instruct"

# --- 2. Define Schemas ---

# Input schema used by both agents
class CityInput(BaseModel):
    city: str = Field(description="The city to get information about.")

# --- 3. Define the Tool (Only for the first agent) ---
def get_weather(city: str) -> str:
    """Retrieves the weather condition of a given city."""
    print(f"\n-- Tool Call: get_weather(city='{city}') --")
    city_weather = {
        "paris": "The weather in Paris is sunny with a temperature of 25 degrees Celsius (77 degrees Fahrenheit).",
        "ottawa": "In Ottawa, it's currently cloudy with a temperature of 18 degrees Celsius (64 degrees Fahrenheit) and a chance of rain.",
        "tokyo": "Tokyo sees humid conditions with a high of 28 degrees Celsius (82 degrees Fahrenheit) and possible rainfall."
    }
    result = city_weather.get(city.strip().lower(), f"Sorry, I don't have weather information in {city}.")
    print(f"-- Tool Result: '{result}' --")
    return result

# --- 4. Configure Agent ---
# Connect to the deployed model by using LiteLlm
api_base_url = os.getenv("LLM_BASE_URL", "http://vllm-llama3-service:8000/v1")
model_name_at_endpoint = os.getenv("MODEL_NAME", "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct")
model = LiteLlm(
    model=model_name_at_endpoint,
    api_base=api_base_url,
)

# Uses a tool and output_key
weather_agent = LlmAgent(
    model=model,
    name="weather_agent_tool",
    description="Retrieves weather in a city using a specific tool.",
    instruction="""You are a helpful agent that provides weather report in a city using a tool.
The user will provide a city name in a JSON format like {"city": "city_name"}.
1. Extract the city name.
2. Use the `get_weather` tool to find the weather. Don't use other tools!
3. Answer on user request based on the weather
""",
    tools=[get_weather],
    input_schema=CityInput,
    output_key="city_weather_tool_result", # Store final text response
)

root_agent = weather_agent

# --- 5. Set up Session Management and Runners ---
session_service = InMemorySessionService()

# Create separate sessions for clarity, though not strictly necessary if context is managed
session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID_TOOL_AGENT)

# Create a runner for EACH agent
weather_runner = Runner(
    agent=weather_agent,
    app_name=APP_NAME,
    session_service=session_service
)

# --- 6. Define Agent Interaction Logic ---
async def call_agent_and_print(
    runner_instance: Runner,
    agent_instance: LlmAgent,
    session_id: str,
    query_json: str
):
    """Sends a query to the specified agent/runner and prints results."""
    print(f"\n>>> Calling Agent: '{agent_instance.name}' | Query: {query_json}")

    user_content = types.Content(role='user', parts=[types.Part(text=query_json)])

    final_response_content = "No final response received."
    async for event in runner_instance.run_async(user_id=USER_ID, session_id=session_id, new_message=user_content):
        # print(f"Event: {event.type}, Author: {event.author}") # Uncomment for detailed logging
        if event.is_final_response() and event.content and event.content.parts:
            # For output_schema, the content is the JSON string itself
            final_response_content = event.content.parts[0].text

    print(f"<<< Agent '{agent_instance.name}' Response: {final_response_content}")

    current_session = session_service.get_session(app_name=APP_NAME,
                                                  user_id=USER_ID,
                                                  session_id=session_id)
    stored_output = current_session.state.get(agent_instance.output_key)

    # Pretty print if the stored output looks like JSON (likely from output_schema)
    print(f"--- Session State ['{agent_instance.output_key}']: ", end="")
    try:
        # Attempt to parse and pretty print if it's JSON
        parsed_output = json.loads(stored_output)
        print(json.dumps(parsed_output, indent=2))
    except (json.JSONDecodeError, TypeError):
         # Otherwise, print as string
        print(stored_output)
    print("-" * 30)


# --- 7. Run Interactions ---
async def main():
    print("--- Testing Agent with Tool ---")
    await call_agent_and_print(weather_runner, weather_agent, SESSION_ID_TOOL_AGENT, '{"city": "Paris"}')
    await call_agent_and_print(weather_runner, weather_agent, SESSION_ID_TOOL_AGENT, '{"city": "Tokyo"}')

if __name__ == "__main__":
    asyncio.run(main())
