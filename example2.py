from LLM import DeepSeek_R1_Distill_Qwen_LLM
import re

# text split function
def split_text(text):
    pattern = re.compile(r'<think>(.*?)</think>(.*)', re.DOTALL) 
    match = pattern.search(text) 
  
    if match: 
        think_content = match.group(1).strip() 
        answer_content = match.group(2).strip() 
    else:
        think_content = "" 
        answer_content = text.strip() 
  
    return think_content, answer_content
  
llm = DeepSeek_R1_Distill_Qwen_LLM(mode_name_or_path = "/root/shared-nvme/llm/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

response = llm("How do I set a goal for learning Python?")
think, answer = split_text(response) 
print(f"{"-"*20}thinking{"-"*20}")
print(think) 
print(f"{"-"*20}answering{"-"*20}")
print(answer) 