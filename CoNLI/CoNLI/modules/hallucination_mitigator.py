import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List
from tqdm import tqdm
from pathlib import Path

import CoNLI.modules.utils.gpt_output_utils as gpt_output_utils
from CoNLI.modules.utils.aoai_utils import AOAIUtil
from CoNLI.modules.hallucination_mitigation_prompt import hallucination_mitigation_prompt
from CoNLI.modules.arguments import OpenaiArguments, MitigationArguments

@dataclass
class HmResult:
    data_id: str
    raw_response: str
    refined_response: str

@dataclass
class HdResult:
    hallucinated_sentence: str # original sentence found in raw_response
    reason: List[str] # why the sentence is hallucination or not
    instruction: str # instruction for rewriting the sentence
    detection_type: str # the detector type that found the hallucination (sentence-level, entity-level)


class HallucinationMitigator :
    def __init__(
            self,
            openai_args : OpenaiArguments = OpenaiArguments(),
            mitigation_args : MitigationArguments = MitigationArguments(),
            config_file: str = (Path(__file__).absolute()).parent.parent/"configs"/"aoai_config.json",
            ) -> None:
        
        self._mitigation_args = mitigation_args
        self._openai_args = openai_args 
        self.aoaiUtil = AOAIUtil(
            config_setting=openai_args.config_setting,
            api_key=self._openai_args.api_key,
            config_file=config_file,
            )
        self._prompt_util = hallucination_mitigation_prompt(use_chat_completions = openai_args.use_chat_completions)

    def mitigate(
            self,
            data_ids: List[str],
            raw_responses: Dict[str, str],
            sources: Dict[str, str],
            hd_results: Dict[str, List[HdResult]],
            ) -> List[HmResult]:
        max_parallelism = self._openai_args.max_parallelism

        items, results = [], []
        for data_id in data_ids:
            raw_response = raw_responses[data_id]
            source = sources[data_id]
            hd_result = hd_results[data_id]

            if len(hd_result) == 0:
                results.append(HmResult(data_id, raw_response, raw_response))
                continue
            
            instructions: str = HallucinationMitigator.get_instructions_by_hd_results(hd_result)

            # sending one request per data
            request = {
                'data_id': data_id,
                'source': source,
                'raw_response': raw_response,
                'rewrite_instructions': instructions,
                }
            items.append(request)
            
        gpt_request_payloads = [
            self.create_payload(
                item = items[i],
                promptUtil = self._prompt_util,
            )
            for i in range(len(items))
        ]

        if len(gpt_request_payloads) > 0:
            gpt_results_raw = list()
            max_workers = min(max(max_parallelism, 1), len(gpt_request_payloads))
            with tqdm(total=len(gpt_request_payloads)) as pbar:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            self.process_payload_by_GPT,
                            payload,
                            self.aoaiUtil,
                            self._openai_args,
                            self._mitigation_args): payload
                        for payload in gpt_request_payloads
                    }
                    
                    for future in as_completed(futures):
                        gpt_results_raw.append(future.result())
                        pbar.update(1)
            results += HallucinationMitigator.parse_gpt_result(gpt_results_raw)
            
        return results
    
    @staticmethod
    def clean_span(x):
        return " ".join(re.sub("[\\<].*?[\\>]", "", x).split())

    @staticmethod
    def get_instructions_by_hd_results(hd_results = List[HdResult]) -> str:
        # dedup sentence
        hd_results_dedup = []
        hd_sentences = set()
        for hd_result in hd_results:
            sentence = HallucinationMitigator.clean_span(hd_result.hallucinated_sentence)
            if sentence not in hd_sentences:
                hd_sentences.add(sentence)
                hd_results_dedup.append(hd_result)

        # get gpt instructions 
        final_instructions = ""
        for i in range(len(hd_results_dedup)):
            final_instructions += \
                f"Rewrite instruction {i+1}:\n" + \
                f"Rrwrite sentence in raw_response: {hd_results[i].hallucinated_sentence}\n" + \
                f"Reason for rewrite the sentence: {hd_results[i].instruction}\n"
        return final_instructions

    @staticmethod
    def create_payload(item, promptUtil : hallucination_mitigation_prompt) -> Dict:
        GPT_OUTPUT_LENGTH_EXPECTATION = 4096 # TODO: make it configurable
        prompt_to_send_to_gpt = promptUtil.create_prompt(
            source = item['source'], 
            raw_response = item['raw_response'], 
            rewrite_instructions = item['rewrite_instructions'], 
            max_tokens = GPT_OUTPUT_LENGTH_EXPECTATION,
            )
        return {'prompt': prompt_to_send_to_gpt, 'item': item}

    # send payload to GPT endpoint and get back the results
    @staticmethod
    def process_payload_by_GPT(payload, aoaiUtil : AOAIUtil, openai_args : OpenaiArguments, mitigation_args : MitigationArguments) -> Dict:

        outputs = []
        try:
            logging.info(f"Start to call GPT to process the data")
            if openai_args.use_chat_completions:
                gpt_response = aoaiUtil.get_chat_completion(
                    messages = payload['prompt'],
                    temperature = mitigation_args.temp,
                    top_p = mitigation_args.top_p, 
                    max_tokens = mitigation_args.max_tokens,
                    frequency_penalty = mitigation_args.freq_penalty,
                    presence_penalty = mitigation_args.presence_penalty,
                    generations=mitigation_args.generations)
                choices = gpt_response['choices']
                for choice in choices:
                    outputs.append(choice['message']['content'])
                payload['gpt_raw_output'] = outputs
            else:
                gpt_response = aoaiUtil.get_completion(
                    prompt = payload['prompt'],
                    max_tokens = mitigation_args.max_tokens,
                    temperature = mitigation_args.temp,
                    top_p = mitigation_args.top_p,
                    frequency_penalty = mitigation_args.freq_penalty,
                    presence_penalty = mitigation_args.presence_penalty,
                    logprobs = mitigation_args.log_prob,
                    generations=mitigation_args.generations)
                choices = gpt_response['choices']
                for choice in choices:
                    outputs.append(choice['text'])
                payload['gpt_raw_output'] = outputs
            logging.info(f"Completed calling GPT to process the data")
        except Exception as exc:
            logging.warning(f"Failed to call GPT: output format wrong!")
            logging.warning(f'Exception: {exc}')
            payload['gpt_raw_output'] = [ 'the format of gpt output is wrong' ]

        return payload

    # Sometimes there is extra text that shows up in the prompt that we want to clean
    # to retrieve the rewritten raw_response
    @staticmethod
    def postprocess_rewrite_result(rewrite_result: str) -> str:
        rewrite_result = rewrite_result.replace('<|im_end|>', '')
        if "Corrected WHOLE CLAIM:" in rewrite_result:
            rewrite_result = rewrite_result.split("Corrected WHOLE CLAIM:",1)[1]
        if "Corrected CLAIM:" in rewrite_result:
            rewrite_result = rewrite_result.split("Corrected CLAIM:",1)[1]
        
        rewrite_result=gpt_output_utils.remove_gpt_output_prefix(rewrite_result)
        rewrite_result = rewrite_result.replace("End WHOLE CLAIM.", "")
        rewrite_result = rewrite_result.replace("End CLAIM", "")
        rewrite_result = rewrite_result.strip()
        return rewrite_result

    # parse gpt result per data
    @staticmethod
    def parse_gpt_result(gpt_results_raw) -> List[HmResult]:
        results = []
        for gpt_result_raw in gpt_results_raw:
            gpt_raw_output = gpt_result_raw["gpt_raw_output"][0]
            results.append(
                HmResult(
                    data_id = gpt_result_raw['item']['data_id'],
                    raw_response = gpt_result_raw['item']['raw_response'],
                    refined_response = HallucinationMitigator.postprocess_rewrite_result(gpt_raw_output),
                )
            )
        return results