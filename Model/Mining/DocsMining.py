import os

from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

from Model.Mining.Common import groupSoupByTag
from Static.Define import path_common, tag

"""
if __name__ == '__main__':
    sentence_df = pd.DataFrame()
    intent_df = pd.DataFrame()
    intent_group_df = pd.DataFrame()
    sentence_index = 0
    intent_index = 0
    intent_group_index = 0
    for root, dirs, files in os.walk(path_common.data.value + "\\Document", topdown=True):
        for file in tqdm(files):
            path = root + "\\" + file
            with open(file=path, mode='r', encoding='utf-8') as f:
                content = f.read()
                html = BeautifulSoup(content.strip("\n"), 'html.parser')
                intent_group_index = intent_group_index + 1
                intent_group = html.h1.text
                intent_group_new = {"intent_group": intent_group, "intent_group_index": intent_group_index}
                intent_group_df = intent_group_df.append(intent_group_new, ignore_index=True)
                groupSoupByTag(html, tag.h2, tag.classify1, tag.sau)
                head = html.find_all(tag.classify1.value)
                for h in head:
                    sub_index = 0
                    elements = h.find_all(recursive=False)
                    h2 = h.h2
                    intent_index = intent_index + 1
                    intent = h2.text
                    intent_new = {'intent_index': intent_index, 'intent': intent, "intent_group": intent_group,
                                  "intent_group_index": intent_group_index}
                    intent_df = intent_df.append(intent_new, ignore_index=True)
                    for e in elements:
                        value = e.text
                        if not value == "":
                            sub_index += 1
                            sentence_index = sentence_index + 1
                            sentence = e.text
                            sentence_new = {'sentence_index': sentence_index, 'intent': intent, 'sentence': sentence,
                                            'intent_index': intent_index, "intent_group": intent_group,
                                            "intent_group_index": intent_group_index, 'sub_i': sub_index}
                            sentence_df = sentence_df.append(sentence_new, ignore_index=True)
        break
    intent_df.to_csv(path_common.intent_list.value, index=False, mode='w')
    sentence_df.to_csv(path_common.sentence_list.value, index=False, mode='w')
    intent_group_df.to_csv(path_common.intent_group_list.value, index=False, mode='w')
    exit()
"""
if __name__ == '__main__':
    answer_df = pd.DataFrame()
    sentence_index = 0
    intent_index = 0
    intent_group_index = 0
    for root, dirs, files in os.walk(path_common.data.value + "\\Document", topdown=True):
        for file in tqdm(files):
            path = root + "\\" + file
            with open(file=path, mode='r', encoding='utf-8') as f:
                content = f.read()
                html = BeautifulSoup(content.strip("\n"), 'html.parser')
                groupSoupByTag(html, tag.h2, tag.classify1, tag.sau)
                head = html.find_all(tag.classify1.value)
                for h in head:
                    sub_index = 0
                    elements = h.find_all(recursive=False)
                    h2 = h.h2
                    intent_index = intent_index + 1
                    intent = h2.text
                    new = {'answer': h, 'intent': intent, 'answer_index': intent_index, 'intent_index': intent_index, }
                    answer_df = answer_df.append(new, ignore_index=True)
        break
    answer_df.to_csv(path_common.answer.value, index=False, mode='w')
    exit()
