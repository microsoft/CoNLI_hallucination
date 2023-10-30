import argparse
from CoNLI.modules.eval.response_quality_evaluation import QualityEvaluator


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--llm_responses',
        required=True,
        help="""The folder containing LLM responses; ; It can also  sentence-level response tsv file with columns:
                DataID, SentenceID, Sentence. The Sentence will be the input strings for detection.""",
        type=str)
    parser.add_argument(
        '--input_src',
        required=True,
        help='The folder where all of your sources are located',
        type=str)
    parser.add_argument(
        '--ground_truth_response',
        required=False,
        default=None,
        help='The folder where all of your ground truth responses are located',
        type=str) 
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    evaluator = QualityEvaluator(args.llm_responses, args.input_src, args.ground_truth_response)
    scores = evaluator.evaluate_responses()
    print("score:", scores)