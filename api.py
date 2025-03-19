from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
import json
import datetime
import torch
import re

# set parameters of your device
DEVICE = "cuda"  # set CUDA
DEVICE_ID = "0"  # set CUDA ID
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE  

# clean memory function of GPU
def torch_gc():
    if torch.cuda.is_available():  # fix CUDA
        with torch.cuda.device(CUDA_DEVICE):  # choose one CUDA device
            torch.cuda.empty_cache()  
            torch.cuda.ipc_collect()  
# text split function
def split_text(text):
    pattern = re.compile(r'<think>(.*?)</think>(.*)', re.DOTALL) # 
    match = pattern.search(text) # matching <think>the process of thought</think>answering
  
    if match: # matching with thought process
        think_content = match.group(1).strip() 
        answer_content = match.group(2).strip() # obtain reply
    else:
        think_content = "" # matching failed
        answer_content = text.strip() # return reply immediatedly
  
    return think_content, answer_content

# create FastAPI application
app = FastAPI()

# process POST request
@app.post("/")
async def create_item(request: Request):
    global model, tokenizer  
    json_post_raw = await request.json()  # get POST request JSON data
    json_post = json.dumps(json_post_raw)  # transfer JSON data to string
    json_post_list = json.loads(json_post)  # transfer sting to Python object
    prompt = json_post_list.get('prompt')  

    messages = [
            {"role": "user", "content": prompt}
    ]

    # use model to get result
    input_ids = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    model_inputs = tokenizer([input_ids], return_tensors="pt").to(model.device)
    generated_ids = model.generate(model_inputs.input_ids,max_new_tokens=8192) 
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    think_content, answer_content = split_text(response) 
    now = datetime.datetime.now()  # get current time
    time = now.strftime("%Y-%m-%d %H:%M:%S")  # transfer format
    # construct JSON
    answer = {
        "response": response,
        "think": think_content,
        "answer": answer_content,
        "status": 200,
        "time": time
    }
    # contruct logs information
    log = f"[{time}], prompt:\"{prompt}\", response:\"{repr(response)}\", think:\"{think_content}\", answer:\"{answer_content}\""
    print(log)  # print logs
    torch_gc()  
    return answer  


if __name__ == '__main__':
    # load model
    model_name_or_path = '/root/shared-nvme/llm/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=CUDA_DEVICE, torch_dtype=torch.bfloat16)

    # run FastAPI
    # use 6006 post 
    uvicorn.run(app, host='0.0.0.0', port=6006, workers=1)  