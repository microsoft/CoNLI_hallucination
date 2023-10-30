import os
from glob import glob
from CoNLI.modules.hallucination_mitigator import HmResult
from CoNLI.modules.data.data_loader import DataLoader
from CoNLI.modules.eval.nlg_metrics import evaluate, calculate_rouge, calculate_bleu, calcualte_bertscore

class QualityEvaluator:
    def __init__(self, hypothesis, sources, gt_response):
        self.dataloader = DataLoader(
            hypothesis=hypothesis,
            sourcefolder=sources
        )
        self.hypothesis = hypothesis
        self.sources = sources
        self.gt_response = gt_response

    def evaluate_responses(self):
        # load response to test against
        data_id_set = set([])
        results = []
        for fname in glob(self.hypothesis.rstrip('/') + "/*.txt"):
            with open(fname, "r") as f:
                data_id = os.path.basename(fname).replace('.txt', '')
                data_id_set.add(data_id)
                results.append(HmResult(data_id, self.dataloader._hypothesis[data_id],f.read().replace("<|im_end|>", "")))

        print('start evaluation')
        if not os.path.isdir(self.gt_response):
            print("No ground truth response provided. Skipping evaluation.")
            return None
        
        # read data
        gt_response, llm_response = [], []        
        gt_response_dict = {}
        for fname in glob(self.gt_response.rstrip('/') + "/*.txt"):
            data_id = os.path.basename(fname).replace('.txt', '')
            if data_id in data_id_set:
                with open(fname, "r") as f:
                    gt_response_dict[data_id] = f.read()

        for result in results:
            gt_response.append(gt_response_dict[result.data_id])
            llm_response.append(result.refined_response)
            
        llm_rouge_scores = evaluate(output_lns=llm_response, reference_lns=gt_response, score_fn=calculate_rouge)
        llm_bleu_scores = evaluate(output_lns=llm_response, reference_lns=gt_response, score_fn=calculate_bleu)
        llm_bertscores = evaluate(output_lns=llm_response, reference_lns=gt_response, score_fn=calcualte_bertscore)
        return {"Rouge":llm_rouge_scores, "Bleu":llm_bleu_scores, "Bertscore":llm_bertscores}