import logging
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_current_weather(city: str) -> str:
    """Get current weather for a city.

    Args:
        city: The city name, e.g., 'Seattle'

    Returns:
        Weather information as a string
    """
    # Mock data
    weather_map = {
        "seattle": "12°C with rainy conditions",
        "san francisco": "18°C with partly cloudy conditions",
        "new york": "22°C with sunny conditions",
        "miami": "28°C with hot and humid conditions",
        "chicago": "15°C with windy conditions",
    }

    city_lower = city.lower()

    if city_lower in weather_map:
        return f"The weather in {city} is currently {weather_map[city_lower]}."
    else:
        return f"Weather data for {city} is not available. Try Seattle, San Francisco, New York, Miami, or Chicago."

# Create the ADK function tool.
logger.debug("Initializing Weather tool")
weather_tool = FunctionTool(get_current_weather)
logger.debug("Weather tool initialized")

# api_base_url = "http://meta-service:8000/v1"
# logger.debug(f"Connecting to vLLM at: {api_base_url}")
RAY_SERVICE_NAME = "llama-31-8b-serve-svc"
RAY_SERVE_PORT = 8000
api_base_url = f"http://{RAY_SERVICE_NAME}:{RAY_SERVE_PORT}/v1"
logger.debug(f"Connecting to Ray Serve application at: {api_base_url}")

root_agent = Agent(
    model=LiteLlm(
        model="hosted_vllm/meta-llama/Llama-3.1-8B-Instruct",
        api_base=api_base_url
    ),
    name="weather_agent",
    instruction="""You are a weather assistant that provides current weather information for different cities.

When asked about the weather in ANY city, immediately use the get_current_weather tool.

Example:
User: "What's the weather like in Seattle?"
<tool:get_current_weather>
{"city": "Seattle"}
</tool>

If the user provides a city not in the database, apologize and suggest they try another major city like Seattle, San Francisco, New York, Miami, or Chicago.""",
    tools=[weather_tool]
)