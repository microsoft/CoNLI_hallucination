import copy
import os
import yaml
from transformers import GPT2TokenizerFast

class hallucination_detection_prompt :
    MAX_TOKEN_8K = 8192
    MAX_TOKEN_32K = 32768

    def __init__(self, use_chat_completions : bool,
                  prompt_resource_root_folder : str = None,
                  max_prompt_tokens : int = MAX_TOKEN_32K
                  ) -> None:
        self._tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self._prompt_resource_root = prompt_resource_root_folder if prompt_resource_root_folder is not None else os.path.join(os.path.dirname(__file__), "..")
        self._max_prompt_tokens = max_prompt_tokens
        self._use_chat_completions = use_chat_completions
        self.prompt = self._load_prompt(use_chat_completions)

    def resolve_file_path(self, file_path : str) -> str:
        # if file_path is not a full path, then resolve it to the full path
        if not os.path.isabs(file_path) :
            file_path = os.path.join(self._prompt_resource_root, file_path)
        return file_path
    
    def load_file_content(self, file_path : str) -> str:
        file_path = self.resolve_file_path(file_path)
        # read all content from the file
        with open(file_path, 'r') as f:
            return f.read()

    def _load_prompt(self, useChatCompletions : bool = False) :
        filename = 'hallucination_detection/generic.nli.v1'
        return self._load_prompt_file(filename=filename, useChatCompletions=useChatCompletions)

    def _load_prompt_file(self, filename : str, useChatCompletions : bool = False) -> str :
        yamlfile = self.resolve_file_path(f'prompts/chat_completions/{filename}.yaml')
        if useChatCompletions:
            return yaml.safe_load(self.load_file_content(yamlfile))
        elif os.path.exists(yamlfile):
            yaml_config = yaml.safe_load(self.load_file_content(yamlfile))
            content = ''
            for item in yaml_config:
                content += item['content'] + '\n'
            return content
        raise FileNotFoundError(f'{yamlfile} was not found')

    def _validate_prompt(self, prompt, max_tokens) :
        if not self._use_chat_completions :
            if len(self._tokenizer(prompt, truncation=True, max_length=32000)['input_ids']) + max_tokens > self._max_prompt_tokens :
                raise ValueError(f'len(prompt) ({len(prompt)}) + max_tokens ({max_tokens}) must be less than {self._max_prompt_tokens}') 
        return prompt

    def _replace(self, promptOrMessageObj, before : str, after : str, simpleReplaceOverride : bool = False) :
        if self._use_chat_completions and not simpleReplaceOverride:
            retval = copy.deepcopy(promptOrMessageObj)
            for item in retval :
                item['content'] = item['content'].replace(before, after)
            return retval
        return promptOrMessageObj.replace(before, after)
    
    
    # The parameter items is a list of dicts with keys: hypothesis, data_id, sentence_id etc.
    # Here the hypothesis in fact is the whole sentence with entity name highlighted
    def create_batch_prompt(self, transcript: str, items: list, max_tokens: int):
        sentence = "\n".join([ "("+str(i)+"). " + item["Hypothesis"] for i, item in enumerate(items)])
        prompt = copy.deepcopy(self.prompt)
        prompt = self._replace(prompt, '{{Source}}', transcript)
        prompt = self._replace(prompt, '{{Hypothesis}}', sentence)
        self._validate_prompt(prompt, max_tokens)
        return prompt
