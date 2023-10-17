import logging


def clean_for_tsv(text):
    return text.replace('\n', ' ').replace('\t', ' ')

def remove_gpt_output_prefix(gpt_out: str) -> str:
    PREFIX='Answer:'
    return gpt_out[gpt_out.find(PREFIX):].replace(PREFIX, '').strip()

def certified_gpt_output_prefix(gpt_out: str) -> bool:
    PREFIX='Answer:\n'
    return PREFIX in gpt_out

def parse_gpt_batch(gpt_out: str, n_item: int):
    gpt_out = remove_gpt_output_prefix(gpt_out)
    ans = []
    for i in range(n_item):
        no = '(' + str(i) + ').'
        next_no = '(' + str(i + 1) + ').'

        item_result = {}
        item_result['IsHallucination'] = False
        item_result['Reason'] = ''
        item_result['Response_Sentence'] = ''

        parse_successful = True
        reason = ''
        q_out = ''
        try:
            if i != (n_item - 1):  # not the last
                q_out = gpt_out.split(no)[1].strip().split(next_no)[0].strip()
            else:
                q_out = gpt_out.split(no)[1].strip()
        except BaseException:
            parse_successful = False

        item_result['Response_Sentence'] = q_out

        if parse_successful:
            if q_out.lower().__contains__('<reason>') and q_out.lower().__contains__('</reason>'):
                reasonSplit = q_out.lower().split('<reason>')
               # item_result['Response_Sentence'] = reasonSplit[0]
                reason = reasonSplit[1].strip().split('</reason>')[0].strip()
            # this is factually correct, so not hallucination
            an = False if ('[c]' in q_out.lower()) else True
            if '[i]' in q_out.lower():
                # this is not factually correct, so hallucination. we weight more on [i] mark
                an = True

            item_result['IsHallucination'] = an
            item_result['Reason'] = reason
        else :
            logging.error(f'Unexpected parsing error seen !!  Sentence returned as non-hallucination ...\nExpectedItemCount:{n_item}\nIter:{i}\n<GPT_OUTPUT>\n{gpt_out}\n</GPT_OUTPUT>')
            item_result['Response_Sentence'] = f'PARSE ERROR SEEN!!! {gpt_out}'

        ans.append(item_result)
    return ans