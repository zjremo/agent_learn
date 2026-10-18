from dotenv import load_dotenv
from agent.reflection import ReflectionAgent
    
load_dotenv(dotenv_path=".reflectionenv")

if __name__ == "__main__":
    reflection_agent = ReflectionAgent(max_iteration=6)
    question = input("请输入你的任务:")
    reflection_agent.run(question=question)