#!/usr/bin/env python

import os
import sys
import json
import importlib
import traceback
import subprocess
import importlib.metadata

from datetime import datetime, timezone

import redhareapiversion
from redhareapi import Kernel

from utils import install_requirements
from utils.prompter import Prompter

dir_user = os.environ['REDHARE_MODEL_PATH']

print(
    f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %z')}: Setting Up ..."
)

# Setup environment variables and paths
deploy_name = os.environ['REDHARE_MODEL_NAME']
dir_user = os.environ['REDHARE_MODEL_PATH']

# Change directory
os.chdir(dir_user)

# Setup working directory
dir_work = f"/opt/wml-edi/work/{deploy_name}"

# Setup python package directory
dir_python_pkg = f"{dir_work}/python_packages"
os.makedirs(dir_python_pkg, exist_ok=True)

# Setup up pip cache directory
dir_pip_cache = f"{dir_python_pkg}/pip_cache"
os.environ['XDG_CACHE_HOME'] = dir_pip_cache
os.environ['PIP_CACHE_HOME'] = dir_pip_cache
os.makedirs(dir_pip_cache, exist_ok=True)

# Install packages and add python install directory into PATH
install_requirements(dir_python_pkg)
sys.path.insert(0, dir_python_pkg)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Import packages
import torch
import tokenizers
import transformers

import accelerate
import bitsandbytes

from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

# Check package versions
print(f"Pytorch version: {importlib.metadata.version('torch')}")
print(f"Tokenizers version: {importlib.metadata.version('tokenizers')}")
print(f"Transformers version: {importlib.metadata.version('transformers')}")
print(f"Accelerate version: {importlib.metadata.version('accelerate')}")
print(f"Bitsandbytes version: {importlib.metadata.version('bitsandbytes')}")
print(f"CUDA version: {torch.version.cuda}")


class MatchKernel(Kernel):

    def on_kernel_start(self, kernel_context):
        Kernel.log_info(
            "on_kernel_start kernel id: " + kernel_context.get_id()
        )

        self.device = "cuda:0"
        self.base_model = "starmpcc/Asclepius-13B"
        self.cache_directory = "/mnts/llm3"

        if os.path.exists(self.cache_directory):
            Kernel.log_info(f"{self.cache_directory} exists")
        else:
            Kernel.log_error(f"{self.cache_directory} does not exist")

        try:
            self.prompter = Prompter("")
            self.tokenizer = LlamaTokenizer.from_pretrained(self.base_model)

            Kernel.log_info(
                "MatchKernel on_kernel_start before Llama.from_pretrained"
            )
            model = LlamaForCausalLM.from_pretrained(
                self.base_model,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=self.cache_directory,
            )
            Kernel.log_info(
                "MatchKernel on_kernel_start after Llama.from_pretrained"
            )

            model.eval()

            self.model = torch.compile(model)
        except Exception as e:
            Kernel.log_error(str(e))

    def on_task_invoke(self, task_context):
        Kernel.log_info("on_task_invoke")
        output_data = {}
        try:
            input_data = json.loads(task_context.get_input_data())

            instruction = input_data.get("instruction", "")
            user_input = input_data.get("input", "")
            parameters = input_data.get("parameters", {})

            temperature = parameters.get("temperature", 0.1)
            top_p = parameters.get("top_p", 0.75)
            top_k = parameters.get("top_k", 40)
            num_beams = parameters.get("num_beams", 1)
            max_new_tokens = parameters.get("max_new_tokens", 256)

            # Hardcode num_beams=1
            num_beams = 1

            # Generate input tokens
            prompt = self.prompter.generate_prompt(
                instruction=instruction, input=user_input
            )
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)

            # Set params
            generation_config = GenerationConfig(
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
            )

            with torch.no_grad():
                generation_output = self.model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                )

            s = generation_output.sequences[0]
            output = self.tokenizer.decode(s)
            output_data["text"] = output

        except Exception as e:
            traceback.print_exc()
            output_data['msg'] = str(e)

        task_context.set_output_data(json.dumps(output_data))

    def on_kernel_shutdown(self):
        pass


if __name__ == '__main__':
    obj_kernel = MatchKernel()
    obj_kernel.run()