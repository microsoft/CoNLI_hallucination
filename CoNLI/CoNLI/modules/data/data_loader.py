from glob import glob
import os
from typing import Dict
import pandas as pd
from CoNLI.modules.data.response_preprocess import ResponsePreprocess


class DataLoader:

    def __init__(self,
                 hypothesis: str,
                 src_folder: str,
                 testmode: int = 0) -> None:

        self._hypothesis = {}
        self._src_docs = {}
        self._hypothesis_preproc_sentences = {}
        self._data_ids = list()

        if os.path.isdir(hypothesis):
            self._hypothesis = self.__load_file_inputs(hypothesis, "*.txt")
            self._hypothesis_preproc_sentences = self.hypothesis_preprocess_into_sentences(
                self._hypothesis)
        elif os.path.isfile(hypothesis) and hypothesis.endswith('.tsv'):
            self._hypothesis_preproc_sentences = self.__load_sentencelevel_file(
                hypothesis)  # hypothesis here is sentences
        else:
            raise ValueError(
                '--inputhypothesis is incorrect. not a valid path or not a tsv file.')

        print(f'Hypotheses Found: {len(self._hypothesis_preproc_sentences)}')

        self._src_docs = self.__load_file_inputs(src_folder, "*.txt")
        print(f'Source Files Found: {len(self._src_docs)}')

        self._data_ids = list(set(self._hypothesis_preproc_sentences.keys(
        )).intersection(set(self._src_docs.keys())))
        print(f'Total Unique Data Found: {len(self._data_ids)}')

        self.__validate()
        self._data_ids = sorted(self._data_ids)

        if testmode > 0:
            self._data_ids = self._data_ids[0:testmode]
            print(
                f'Running reduced dataset of {testmode} IDs: {self._data_ids} ...')

    def __load_file_inputs(self, folderpath: str,
                           searchpattern: str = "") -> Dict[str, str]:
        retval = {}
        for fname in glob(os.path.join(folderpath, searchpattern)):
            with open(fname, "r", encoding="utf-8") as f:
                fnamenoext = os.path.splitext(os.path.basename(fname))[0]
                retval[fnamenoext] = f.read()
        return retval

    def __load_sentencelevel_file(self, hypothesisfile) -> dict:
        hypdf = pd.read_csv(hypothesisfile, sep='\t', header=0)
        hypsens = {}
        important_columns = hypdf[["DataID", "SentenceID", "Sentence"]]
        important_columns = important_columns.drop_duplicates()
        for index, row in important_columns.iterrows():
            # because the loaded result IDs are int, but source IDs are string
            En_Id = str(row["DataID"])
            if not hypsens.__contains__(En_Id):
                hypsens[En_Id] = []
            sen_dict = {
                'sentence_id': row["SentenceID"],
                'text': row["Sentence"]
            }
            hypsens[En_Id].append(sen_dict)
        return hypsens

    def __validate(self) -> None:
        if len(self._data_ids) == 0:
            print('=============\n!! ERROR !!\n=============\nThere are no matching data.\nTerminating the run..\n=============')
            raise ValueError(
                '--inputhypothesis or --inputsource is incorrect.  We found no matching data ids.')

        if len(self._data_ids) != len(self._hypothesis_preproc_sentences) or len(self._data_ids) != len(self._src_docs):
            print('=============\n!! WARNING !!\n=============\nThe number of unique data ids is different between your source and hypothesis folders.\nPlease confirm your inputs are correct before proceeding...\n=============')

    def hypothesis_preprocess_into_sentences(self, hypothesis) -> dict:
        hyp_sentences_preproc = {}
        # Configure response preprocessing module
        # There is an expectation that the input data is cleaned before running Hallucination Detection
        preprocess_skipStartsWithSet = set(['#'])
        preprocess_replaceWordSet = set([])
        responsepreprocess = ResponsePreprocess(
            skip_starts_with_set=preprocess_skipStartsWithSet,
            replace_set=preprocess_replaceWordSet)
        for id in hypothesis.keys():
            hyp_sentences_preproc[id] = responsepreprocess.Preprocess(
                hypothesis[id])
        return hyp_sentences_preproc
