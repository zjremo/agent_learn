from dotenv import load_dotenv
from agent.react import OpenAICompletionClient, AGENT_SYSTEM_PROMPT, available_tools
import os
import re

load_dotenv(dotenv_path='.reactenv')

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
REACT_MAX_LOOP = int(os.getenv("REACT_MAX_LOOP"))

# 1. create LLM client
llm_client = OpenAICompletionClient(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# 2. create prompt and history
user_prompt = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
prompt_history = [
    f"user: {user_prompt}"
]

print(f"用户输入: {user_prompt}\n" + "="*40)

# 3. React agent loop
for i in range(REACT_MAX_LOOP):
    print(f"\n\n===== Iteration {i+1} =====\n")

    # 3.1 build full prompt
    full_prompt = "\n".join(prompt_history)

    # 3.2 generate response
    llm_response = llm_client.generate(prompt=full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
    # 模型可能会输出多余的Thought-Action，需要截断
    match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_response, re.DOTALL)
    if match:
        truncated = match.group(1).strip()
        if truncated != llm_response.strip():
            llm_response = truncated
            print("已截断多余的 Thought-Action 对")
    print(f"模型输出:\n{llm_response}\n")
    prompt_history.append(llm_response)

    # 3.3 execute action
    action_match = re.search(r'Action: (.*)', llm_response)
    if not action_match:
        observation = "错误: 未能解析到 Action 字段。请确保你的回复严格遵循 'Thought: ... Action: ...' 的格式。"
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "="*40)
        prompt_history.append(observation_str)
        continue

    action_str = action_match.group(1).strip()

    if action_str.startswith("Finish"):
        final_answer = re.match(r"Finish\[(.*?)\]", action_str)[1]
        print(f"最终答案: {final_answer}\n" + "="*40)
        break

    tool_name = re.search(r"(\w+)\(", action_str).group(1)
    args_str = re.search(r"\((.*)\)", action_str).group(1)
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))

    if tool_name in available_tools:
        observation = available_tools[tool_name](**kwargs)
    else:
        observation = f"错误：未定义的工具 '{tool_name}'"

    # 3.4. 记录观察结果
    observation_str = f"Observation: {observation}"
    print(f"{observation_str}\n" + "="*40)
    prompt_history.append(observation_str)
