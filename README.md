## llmdeploy
There are several basic operations for DeepSeek-R1-Distill-Qwen-7B.

### 1. Introduction & Installation
This project demonstrates four approaches for deploying and invoking large models:

**FastAPI Deployment**: Build an API service using FastAPI to enable online calls to large models.

**Langchain Integration**: Integrate Langchain to support chain-of-thought conversations and task processing.

**WebDemo Deployment**: Provide a simple web-based demo for users to experience and test the model's capabilities.

**vLLM Deployment**: Utilize vLLM to deliver high-performance large model services, optimizing response speed and resource usage.

We will present the full implementation of each function in the following sections. Code has been tested with:
```
----------------
ubuntu 22.04
python 3.12
cuda 12.1
pytorch 2.2.0
----------------
```
Please use the following command for installation.
```bash
# It is recommended to create a new environment, we named it llm.
conda create -n llm python==3.12
conda activate llm

# upgrade pip
python -m pip install --upgrade pip

# Replace the installation of the pypi source accelerator
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install packages and other dependencies
pip install -r requirements.txt

```
Next, we need to download the module. Create a new python file named model_download.py in your project.
```bash
# The code of this python file should be setted as follows(remember change the cache_dir with your own catagory):

from modelscope import snapshot_download

model_dir = snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', cache_dir='/root/shared-nvme/llm', revision='master')
```

And now run this python file.
```bash
# run model_download.py
python model_download.py
```

### 2. FastAPI Deployment

Change the model_name_or_path in api.py to your own path and run api.py to work out your service. The loading is successful if the following information is displayed:
![](assets/pic1.png)


The default deployment is on port 6006, which is called through curl as follows:
```bash
curl -X POST "http://127.0.0.1:6006" \
     -H 'Content-Type: application/json' \
     -d '{"prompt": "Please explain briefly how to put an elephant in the refrigerator."}'
```
After this, you will get the final result. You can also make calls using the requests library in python. We also provide example1.py to do this operation.
### 3. Langchain Integration
To streamline the development of LLM applications, we integrate locally deployed DeepSeek_R1_Distill_Qwen_LLM into LangChain by creating a custom LLM class. This custom class inherits from LangChain.llms.base.LLM and overrides the constructor and the _call function. The _call function is responsible for taking a prompt, processing it with the DeepSeek model, and returning the model’s generated response. With this approach, you can interact with your DeepSeek model using the same interface as any other LangChain LLM.




### 4. WebDemo Deployment

### 5. vLLM Deployment
