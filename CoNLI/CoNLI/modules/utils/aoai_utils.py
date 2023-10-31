import datetime
import json
import re
from azure.identity import AzureCliCredential, ManagedIdentityCredential
import logging
import os
import openai
import time
from pathlib import Path

from CoNLI.modules.utils.gpt_output_utils import certified_gpt_output_prefix

class AOAIUtil:

    def __init__(
            self,
            config_setting: str = "gpt-4-32k",
            config_file: str = (Path(__file__).absolute()).parent.parent/"configs"/"aoai_config.json") -> None:
        self.auth_token = None
        self.default_credential = None
        self.config_setting = config_setting
        with open(config_file, "r") as config_file:
            config = json.load(config_file)
        self.default_engine = str(config[config_setting]["DEFAULT_ENGINE"])
        openai.api_type = str(config[config_setting]["API_TYPE"])
        openai.api_base = str(config[config_setting]["OPENAI_API_BASE"])
        openai.api_version = str(config[config_setting]["OPENAI_API_VERSION"])

        if "OPENAI_API_KEY" in config[config_setting]:
            openai.api_key = str(config[config_setting]["OPENAI_API_KEY"])
        else:
            raise Exception("Please config OPENAI API KEY in aoai_config.json")

    def get_engine(self, engine: str = None) -> str:
        if engine:
            return engine
        else:
            return self.default_engine

    def get_completion(
            self,
            prompt: str,
            max_tokens: int,
            temperature: float,
            top_p: float,
            engine: str = None,
            frequency_penalty: float = 0,
            presence_penalty: float = 0,
            logprobs: int = None,
            stop: list() = ["<|im_end|>"],
            generations: int = 1,
            should_retry: bool = True):
        while True:
            try:
                response = openai.Completion.create(
                    engine=self.get_engine(engine),
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    logprobs=logprobs,
                    stop=stop,
                    n=generations)
                if not certified_gpt_output_prefix(response['choices'][0]['text']):
                    raise Exception('GPT undesired output due to rpm limit reached, resending current request')
                break
            except Exception as e:
                errStr = str(e).lower()
                if should_retry and (
                        "rate limit" in errStr or "server is currently overloaded" in errStr or "server is overloaded" in errStr):
                    logging.info("Retrying after rate limit error")
                    time.sleep(5)
                    continue
                elif should_retry and ("no healthy upstream" in errStr or "error communicating with openai" in errStr):
                    logging.info(f'Unexpected, retryable error: {errStr}')
                    time.sleep(5)
                    continue
                else:
                    logging.error(f'Unexpected, unrecoverable error: {errStr}')
                    return None
        return response

    def get_chat_completion(
            self,
            messages,
            engine: str = None,
            temperature: float = 0.0,
            top_p: float = 0.0,
            max_tokens: int = 50,
            frequency_penalty: float = 0,
            presence_penalty: float = 0,
            generations: int = 1,
            stop: list() = ["<|im_end|>"],
            max_retry_count: int = 10):

        retry_count = 0
        while True:
            if retry_count > max_retry_count:
                logging.error(f"Max retry count exceeded, aborting")
                return None
            try:
                response = openai.ChatCompletion.create(
                    engine=self.get_engine(engine),
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                    n=generations)
                raw_output = response['choices'][0]['message']['content']
                if not certified_gpt_output_prefix(raw_output):
                    raise Exception(f'GPT undesired output due to rpm limit reached, resending current request. \n<GPT_OUTPUT>\n{raw_output}\n</GPT_OUTPUT>')
                break
            except Exception as e:
                already_slept = False
                errStr = str(e).lower()
                if 'rate limit' in errStr or 'overloaded with other requests' in errStr:
                    logging.warning(f"Retrying after rate limit error, retry count: {retry_count}")
                    retry_count += 1
                    for w in errStr.split(' '):
                        if w.isdigit():
                            time.sleep(int(w))
                            already_slept = True
                            break
                    if not already_slept:
                        time.sleep(5)
                    continue
                elif 'unauthorized' in errStr:
                    logging.error(f'Unauthorized error seen: {errStr}')
                    time.sleep(5)
                    retry_count += 1
                    continue
                # This error means that content filtering is on, it is not a
                # recoverable error
                elif 'please modify your prompt and retry' in errStr:
                    logging.error(f'Unexpected, unrecoverable error: {errStr}')
                    return None
                elif 'please reduce the length of the messages' in errStr:
                    logging.error(f'Unrecoverable error - input too long: {errStr}')
                    return None
                elif 'invalid subscription key' in errStr or 'wrong api endpoint' in errStr:
                    logging.error(f'Unrecoverable error - access denied: {errStr}')
                    return None
                else:
                    logging.error(f'Unexpected, retryable error: {errStr}. retry count: {retry_count}')
                    time.sleep(5)
                    retry_count += 1
                    continue
        return response

    def convert_to_chat_format(self, text: str) -> str:
        reg_str = "<\|im_start\|>(.*?)<\|im_end\|>"
        res = re.findall(reg_str, text, flags=re.DOTALL)
        chat = []
        for turn in res:
            role, content = turn.split("\n", 1)
            t = {"role": role, "content": content}
            chat.append(t)
        return chat
    
    @staticmethod
    def get_model_context_length(config_file : str, config_setting : str) -> int:
        with open(config_file, "r") as config_file:
            config = json.load(config_file)
        
        setcion = config[config_setting]
        key_name = "MAX_CONTEXT_LENGTH"
        if key_name in setcion:
            return int(setcion[key_name])
        else :
            logging.warning(f'Key {key_name} not found in config file for {config_setting}, using default value of 8K')
            return 8192 #8K as default for GPT


