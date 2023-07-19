#!/usr/bin/env python

import os
import sys
import json
import importlib
import traceback
import subprocess

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
import transformers

import accelerate
import bitsandbytes

from peft import PeftModel
from transformers import (
    GenerationConfig, LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
)

print(f"CUDA version: {torch.version.cuda}")


class MatchKernel(Kernel):

    def on_kernel_start(self, kernel_context):
        Kernel.log_info(
            "on_kernel_start kernel id: " + kernel_context.get_id()
        )

        self.device = "cuda:0"
        self.base_model = "yahma/llama-7b-hf"
        self.lora_weights = "yahma/alpaca-7b-lora"
        self.cache_directory = "/mnts/llm"

        if os.path.exists(self.cache_directory):
            Kernel.log_info(f"{self.cache_directory} exists")
        else:
            Kernel.log_error(f"{self.cache_directory} does not exist")

        try:
            self.prompter = Prompter("")
            self.tokenizer = LlamaTokenizer.from_pretrained(self.base_model)

            Kernel.log_info(
                "MatchKernel on_kernel_start before from_pretrained"
            )
            model = LlamaForCausalLM.from_pretrained(
                self.base_model,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map="auto",
                cache_dir=self.cache_directory,
            )
            Kernel.log_info(
                "MatchKernel on_kernel_start after from_pretrained"
            )

            Kernel.log_info(
                "MatchKernel on_kernel_start before from_pretrained"
            )
            model = PeftModel.from_pretrained(model, self.lora_weights)
            Kernel.log_info(
                "MatchKernel on_kernel_start after from_pretrained"
            )

            model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
            model.config.bos_token_id = 1
            model.config.eos_token_id = 2

            self.model = torch.compile(model)
        except Exception as e:
            Kernel.log_error(str(e))

    def on_task_invoke(self, task_context):
        Kernel.log_info("on_task_invoke")
        output_data = {}
        try:
            input_data = json.loads(task_context.get_input_data())

            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf
            # asdfasdfasdf

        except Exception as e:
            traceback.print_exc()
            output_data['msg'] = str(e)
            task_context.set_output_data(json.dumps(output_data))

    def on_kernel_shutdown(self):
        pass


if __name__ == '__main__':
    obj_kernel = MatchKernel()
    obj_kernel.run()