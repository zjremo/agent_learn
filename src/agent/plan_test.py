from dotenv import load_dotenv
import os

from agent.plan_and_solve import PlanAndSolveAgent, LLMClient, Planner, Solver

try:
    load_dotenv(dotenv_path=".planenv")
except FileNotFoundError:
    print("Error: .planenv file not found")
    exit(1)
except Exception as e:
    print(f"Error loading .planenv: {e}")
    exit(1)

# read the env variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

if __name__ == "__main__":
    try:
        question = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
        llm_client = LLMClient(model_name=OPENAI_MODEL, api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        planner = Planner(llm_client=llm_client)
        solver = Solver(llm_client=llm_client)
        agent = PlanAndSolveAgent(planner, solver)
        agent.run(question)
    except Exception as e:
        print(e)