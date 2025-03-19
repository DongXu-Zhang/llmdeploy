# vllm_model.py
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import json

# Specify modelscope when automatically downloading models; Otherwise, it will download from HuggingFace
os.environ['VLLM_USE_MODELSCOPE']='True'

def get_completion(prompts, model, tokenizer=None, max_tokens=8192, temperature=0.6, top_p=0.95, max_model_len=2048):
    stop_token_ids = [151329, 151336, 151338]
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop_token_ids=stop_token_ids)
    # Initialize the vLLM inference engine
    llm = LLM(model=model, tokenizer=tokenizer, max_model_len=max_model_len,trust_remote_code=True)
    outputs = llm.generate(prompts, sampling_params)
    return outputs


if __name__ == "__main__":    
    model='/root/shared-nvme/llm/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B' # the path of model
    # model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    tokenizer = None

    
    text = ["Please help me develop a short Python learning plan for beginners<think>\n", ] 

    outputs = get_completion(text, model, tokenizer=tokenizer, max_tokens=8192, temperature=0.6, top_p=0.95, max_model_len=2048) # 思考需要输出更多的 Token 数，max_tokens 设为 8K，根据 DeepSeek 官方的建议，temperature应在 0.5-0.7，推荐 0.6

    # print output
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        if r"</think>" in generated_text:
            think_content, answer_content = generated_text.split(r"</think>")
        else:
            think_content = ""
            answer_content = generated_text
        print(f"Prompt: {prompt!r}, Think: {think_content!r}, Answer: {answer_content!r}")