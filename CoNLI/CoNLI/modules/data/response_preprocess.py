import nltk
from typing import Dict, List


class ResponsePreprocess:

    def __init__(self, skip_starts_with_set, replace_set) -> None:
        nltk.download('punkt')
        self._break_sentence = nltk.data.load(
            "tokenizers/punkt/english.pickle")
        self._skip_starts_with_set = skip_starts_with_set
        self._replace_set = replace_set

    def preprocess(self, text: str) -> List[Dict[str,str]]:
        retval = list()
        shouldSkip = False
        sentencesRemoved = 0
        sentenceId = 0
        for line in text.split('\n'):
            for sent in self._break_sentence.tokenize(line):
                sent = sent.replace('__lf1__', '').replace('__lf2__', '')
                sent = sent.strip()

                if (sent.startswith('#') and sent.endswith('#')) or (sent.isupper()):
                    sentencesRemoved += 1
                    continue

                for swWord in self._skip_starts_with_set:
                    if sent.startswith(swWord):
                        shouldSkip = True
                        continue

                if shouldSkip:
                    shouldSkip = False
                    sentencesRemoved += 1
                    continue

                for rWord in self._replace_set:
                    sent = sent.replace(rWord, '')

                sent = sent.replace('<|im_end|>','').strip()

                if len(sent) <= 3:
                    sentencesRemoved += 1
                    continue
                sentenceId += 1
                retval.append({'sentence_id': sentenceId, 'text': sent})
        # print(f'Sentences Removed: {sentencesRemoved}')
        return retval
