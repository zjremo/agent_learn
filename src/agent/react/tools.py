import os
import requests

from tavily import TavilyClient

def get_weather(city: str) -> str:
    """
    通过调用 wttr.in API 查询真实的天气信息。
    """
    # API端点，我们请求JSON格式的数据
    url = f"https://wttr.in/{city}?format=j1"

    try:
        resp = requests.get(url=url)
        resp.raise_for_status()
        data = resp.json()

        # 提取天气信息
        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value']
        temp = current_condition['temp_C']

        # 格式化为Observation 自然语言
        return f"The weather in {city} is {weather_desc} and the temperature is {temp} degrees Celsius."
    except requests.exceptions.RequestException as e:
        return f"Error accessing weather API: {e}"
    except (KeyError, IndexError) as e:
        return f"Error parsing weather data: {e}"

def get_attraction(city: str, weather: str) -> str:
    """
    根据城市和天气搜索推荐的旅游景点。
    """
    # Step1: 获取Tavily API密钥
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        return "Error: TAVILY_API_KEY is not set"
    
    # Step2: 创建Tavily客户端
    client = TavilyClient(api_key=tavily_api_key)

    # Step3: 构建查询
    query = f"'{city}' 在 '{weather}' 天气下的推荐景点及理由"

    try:
        # Step4: 执行查询
        resp = client.search(query=query, search_depth="basic", include_answer=True)

        if resp.get("answer"):
            return resp["answer"]
        
        formatted_results = []
        for result in resp.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")
        
        if not formatted_results:
            return "No results found"
        
        return "Here are some attractions you can visit in {city} in {weather}:\n" + "\n".join(formatted_results)
    except Exception as e:
        return f"Error accessing Tavily API: {e}"

available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction
}