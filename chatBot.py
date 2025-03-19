from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st
import re

# Create a title and a link in the sidebar
with st.sidebar:
    st.markdown("## DeepSeek-R1-Distill-Qwen-7B LLM")
    "[Open source large model eating guide self-llm](https://github.com/datawhalechina/self-llm.git)"
    max_length = st.slider("max_length", 0, 8192, 8192, step=1)

# Create a title and a subtitle
st.title("💬 DeepSeek R1 Distill Chatbot")
st.caption("🚀 A streamlit chatbot powered by Self-LLM")

# define the path of the model
mode_name_or_path = '/root/shared-nvme/llm/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'

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

# define a function to get the model and tokenizer
@st.cache_resource
def get_model():
    # get tokenizer from pretrained model
    tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # get the model from the pre-trained model and set the model parameters
    model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, torch_dtype=torch.bfloat16,  device_map="auto")

    return tokenizer, model

# loading model and tokenizer of Qwen2.5
tokenizer, model = get_model()

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "What can I help you?"}]

# Iterate over all messages in session_state and display them on the chat interface
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    input_ids = tokenizer.apply_chat_template(st.session_state.messages,tokenize=False,add_generation_prompt=True)
    model_inputs = tokenizer([input_ids], return_tensors="pt").to('cuda')
    generated_ids = model.generate(model_inputs.input_ids,max_new_tokens=max_length)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    think_content, answer_content = split_text(response) 
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.expander("the process of model thought"):
        st.write(think_content) 
    st.chat_message("assistant").write(answer_content) 
    # print(st.session_state) 