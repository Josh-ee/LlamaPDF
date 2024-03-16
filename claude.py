import anthropic

from dotenv import load_dotenv
import os

load_dotenv()  # This loads the variables from .env

api_key = os.getenv('ANTHROPIC_API_KEY')


client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=api_key,
)

message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    temperature=0.0,
    system="You are a sarcastic AI.",
    stream=True,
    messages=[
        {"role": "user", "content": "How are you today?"}
    ]
)

# print(vars(message))

# for text in message:
#       print(text, end="", flush=True)
print(message.content)