from openai import OpenAI

class OpenAICompletionClient:
    def __init__(self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    def generate(self, prompt: str, system_prompt: str) -> str:
        print('Generating response...')
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            answer =resp.choices[0].message.content
            print(f'LLM response successfully generated')
            return answer
        except Exception as e:
            print(f'Error generating response: {e}')
            return "Error: Failed to generate response"