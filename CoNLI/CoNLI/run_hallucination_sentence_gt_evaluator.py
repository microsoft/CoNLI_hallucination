import argparse
import pandas as pd
import os
from sklearn.metrics import classification_report
class SentenceLevelEvaluator:

    def __init__(self, args, gtfile, hdfile) -> None:
        self._gt_df = pd.read_csv(
            gtfile, sep='\t', encoding='ISO-8859-1', header=0)
        self._hd_df = pd.read_csv(
            hdfile, sep='\t', encoding='ISO-8859-1', header=0)

        self._filter_label_append = ''
        filter = args.filter

        if len(filter) > 0:
            before_len = len(self._gt_df)
            filter_split = filter.split('=')
            filter_col = filter_split[0]
            filter_value = filter_split[1]

            if filter_col not in set(self._gt_df.columns):
                raise ValueError(
                    f'Column [{filter_col}] not found in {gtfile}')

            self._filter_label_append = f'-{filter_col}-{filter_value}'
            print(self._filter_label_append)

            self._gt_df = self._gt_df[self._gt_df[filter_col] == filter_value]
            after_len = len(self._gt_df)
            if after_len == 0:
                raise ValueError(
                    f'Value [{filter_value}] not located in column [{filter_col}] in {gtfile}')

            print(
                f'GT File filtered down to {filter_col} == {filter_value} - Before: {before_len} After: {after_len}')

    def __Run_Analysis_and_Save_Output(self, df_prediction, output_folder, label):
        label = f'{label}{self._filter_label_append}'
        outputtextfile = os.path.join(
            output_folder, f'intermediate/Analysis.Results.{label}.txt')
        if not df_prediction.empty:
            pred_gt = pd.merge(
                df_prediction, 
                self._gt_df, 
                how = 'right', 
                left_on = ['DataID', 'SentenceID'], 
                right_on = ['DataID', 'SentenceID'])
        else:
            pred_gt = self._gt_df.copy()
            pred_gt['IsHallucinatedSentence'] = False
            pred_gt['reason'] = "Not Detected"
            pred_gt['detectiontype'] = label
        
        pred_gt.fillna(False, inplace=True)

        val_dict = {True: 1, False: 0, 'NaN': 0}
        pred_gt['IsHallucination_pred'] = pred_gt['IsHallucinatedSentence'].map(
            val_dict).fillna(0).astype(int)
        target_names = [
            f'CorrectSentences-{label}',
            f'HallucinatedSentences-{label}']

        outputstr = classification_report(
            pred_gt['IsHallucination'],
            pred_gt['IsHallucination_pred'],
            target_names=target_names,
            digits=4)
        print(outputstr)
        with open(outputtextfile, 'w') as outF:
            outF.write(outputstr)

        if label == 'OVERALL':
            incorrect = pred_gt[pred_gt['IsHallucination'] != pred_gt['IsHallucination_pred']][['DataID', 'SentenceID', 'Sentence', 'IsHallucination','IsHallucination_pred', 'reason']]
            incorrectfile = os.path.join(output_folder, 'intermediate/incorrect_classification.tsv')
            incorrect.to_csv(incorrectfile, sep="\t", index=False)

    def Print_Sentence_Level_Results(self, args):
        output_folder = args.output_folder
        detectiontypes = list(self._hd_df['detectiontype'].unique())
        detectiontypes.append('OVERALL')

        for detectiontype in detectiontypes:
            # Select
            tmp_df = self._hd_df[['data_id', 'sentenceid', 'detectiontype', 'reason']]

            # Filter
            if detectiontype != 'OVERALL':
                tmp_df = tmp_df[tmp_df['detectiontype'] == detectiontype]

            # Drop
            tmp_df = tmp_df.rename(columns={
                'data_id': 'DataID',
                'sentenceid': 'SentenceID'
            })
            
            tmp_df = tmp_df.groupby(['DataID', 'SentenceID']).agg(list)

            # Assign
            tmp_df = tmp_df.assign(IsHallucinatedSentence=True)
            self.__Run_Analysis_and_Save_Output(
                tmp_df, output_folder, detectiontype)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gtfile',
        required=True,
        help='The groundtruth sentence level tsv file with  columns: DataID, SentenceID,  Sentence, IsHallucination.',
        type=str)
    parser.add_argument(
        '--hd_result_folder',
        required=True,
        help='The output folder where you ran your hallucinations',
        type=str)
    parser.add_argument(
        '--filter',
        default='',
        help='A single additional filter pivot (eg. Severity=critical)',
        type=str)
    parser.add_argument(
        '--output_folder',
        required=True,
        help='Directory where the analysis output will be written.',
        type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_arguments()
    args.output_folder = args.output_folder.rstrip('/')
    print(f'Input Arguments: {args}')

    gtfile = args.gtfile
    hdfile = os.path.join(
        args.hd_result_folder,
        "intermediate/HallucinationFinal.tsv")
    os.makedirs(os.path.join(args.output_folder, "intermediate"), exist_ok=True)

    evaluator = SentenceLevelEvaluator(args, gtfile, hdfile)
    evaluator.Print_Sentence_Level_Results(args)
