from llm import chat_completion
from tasks import task_dictionary, task_prompt_segment
from utils import console

from .prompts import system, examples, user
from . import verify_plan

def agent(query):
    completion = chat_completion(
        system(task_prompt_segment),
        examples(), 
        user(query)
    )
    
    # Extract the plan from the response
    if completion and "choices" in completion:
        plan = completion["choices"][0]["message"]["content"]
    else:
        console.error("No valid response from completion")
        return None

    print("Got Plan")
    print("========")
    print(plan)
    print("========")

    print("Is valid:", verify_plan(plan, task_dictionary))

    return plan