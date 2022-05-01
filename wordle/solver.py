import numpy as np
import re
from enum import Enum, auto

import pandas as pd
from sparrow.performance import MeasureTime
from pypinyin import pinyin, lazy_pinyin, Style
from typing import List, Set
from pypinyin import pinyin_dict, phonetic_symbol
# from pypinyin.style._constants import _FINALS, _INITIALS
# from pypinyin.style.finals import Style, to_finals_tone3, to_finals_tone, to_finals_tone2
from copy import copy, deepcopy
from pypinyin.style._utils import get_initials, get_finals
from pyprobar import probar
import math
from sparrow import save, load
from sparrow import rel_to_abs
from sparrow.functions import topk
from sparrow.performance import MeasureTime
from pprint import pprint
from wordle.utils import *


def exclude_idx(a: list, idx):
    res = []
    for index, i in enumerate(a):
        if index == idx:
            continue
        res.append(i)
    return res


def exclude_range_idx(start, end, idx):
    res = []
    for index in range(start, end):
        if index == idx:
            continue
        res.append(index)
    return res


class WordleSolver:
    def __init__(self, use_cache=True):
        self.idom_content = self.read_idiom(idx_end=2000)
        self.all_idiom_phone_list = self.get_all_idiom_phone_list(use_cache=use_cache)
        self.all_phone = self.get_all_phones()
        self.shuffled_all_idiom_array = np.array(self.all_idiom_phone_list)
        np.random.shuffle(self.shuffled_all_idiom_array)

    def get_all_idiom_phone_list(self, use_cache=True, cached_filename=rel_to_abs("phone_list")):
        if use_cache:
            return load(cached_filename)
        else:
            idiom_phone_list = self.separate_to_phone(self.idom_content)
            save(cached_filename, idiom_phone_list)
            return idiom_phone_list

    @staticmethod
    def read_idiom(filename: str = rel_to_abs("../datasets/words_freq.csv"), idx_start=3, idx_end=22621):
        df: pd.DataFrame = pd.read_csv(filename, encoding="utf-8")
        return df['word'].tolist()[idx_start: idx_end]

    @staticmethod
    def separate_to_phone(idiom_list):
        """Resturn:
            [
                {"phone": [{(声母, 0)，(韵母, 1), (音调, 2)},{}, {}, {}],
                "word": word},
                {...},
                ...
            ]
        """
        result = []
        for word in idiom_list:
            py_list = pinyin(word, style=Style.TONE3, neutral_tone_with_five=True)
            strict = False
            phone_list = [
                {(get_initials(i[0], False), 0), (get_finals(i[0], strict)[:-1], 1), (get_finals(i[0], strict)[-1], 2)}
                for i in py_list]

            result.append({"phone": phone_list, "word": word})

        return result

    def get_all_phones(self):
        word_bag = set()
        for item in self.all_idiom_phone_list:
            [word_bag.add(j) for i in item['phone'] for j in i]
        return word_bag

    @staticmethod
    def compare_word_phone(candi_list: List[Set], target_list: List[Set]) -> List[List[Set]]:
        """Ruturn [[{here}, {other}, {not exist}], [..], [..], [..]]
        """
        # phone_candi, phone_target = deepcopy(candi_list), deepcopy(target_list)
        phone_candi, phone_target = [copy(i) for i in candi_list], [copy(i) for i in target_list]
        compare_res = [[set(), set(), set()] for _ in range(4)]

        #  method 1
        for idx, candi_set, target_set in zip(range(len(phone_target)), phone_candi, phone_target):
            i_intersect = candi_set & target_set
            compare_res[idx][0] |= i_intersect
            candi_set -= i_intersect
            target_set -= i_intersect

        for idx_candi in range(len(phone_candi)):
            for idx_target in range(len(phone_target)):
                i_intersect = phone_candi[idx_candi] & phone_target[idx_target]
                phone_candi[idx_candi] -= i_intersect
                phone_target[idx_target] -= i_intersect
                compare_res[idx_candi][1] = compare_res[idx_candi][1] | i_intersect
            compare_res[idx_candi][2] = phone_candi[idx_candi]

        # method 2  has problem
        # for idx_candi in range(len(phone_candi)):
        #     for idx_target in [idx_candi, *exclude_range_idx(0, 4, idx_candi)]:
        #         i_intersect = phone_candi[idx_candi] & phone_target[idx_target]
        #         if idx_candi == idx_target:
        #             compare_res[idx_candi][0] = compare_res[idx_candi][0] | i_intersect
        #             phone_candi[idx_candi] -= i_intersect
        #             phone_target[idx_target] -= i_intersect
        #         else:
        #             compare_res[idx_candi][1] = compare_res[idx_candi][1] | i_intersect
        #             phone_candi[idx_candi] -= i_intersect
        #     compare_res[idx_candi][2] = phone_candi[idx_candi]

        return compare_res

    def expand_diff_info(self, diff_info: List[List[Set]]):
        expanded_info = deepcopy([i[0] for i in diff_info])
        not_exist_set = set()
        for i in range(len(diff_info)):
            not_exist_set |= diff_info[i][2]
        all_phone = self.all_phone - not_exist_set
        for idx1 in range(len(diff_info)):
            here, other = expanded_info[idx1], diff_info[idx1][1]
            here_options = all_phone - other
            self.merge_set(here, here_options, inplace=True)
            # 这里还没考虑清楚对于 `它处`的音在它处如何安放
        return expanded_info

    @staticmethod
    def merge_set(set1, set2, inplace=True):
        """merge set2 to set1 accordding to Set[Tuple[1]]"""
        key_list = [i[1] for i in set1]
        res_set = set1 if inplace else deepcopy(set1)
        for i in set2:
            if i[1] not in key_list:
                res_set.add(i)
        return res_set

    @staticmethod
    def match(target_list: List[Set], expanded_diff_info: List[Set]):
        for i, j in zip(target_list, expanded_diff_info):
            if not i.issubset(j):
                return False
        return True

    def calc_candidate_counts(self, expanded_info, candidate_words_list):
        count = 0
        for i in candidate_words_list:
            if self.match(i['phone'], expanded_info):
                count += 1
        return count

    def word_to_human_diff(self, candidate_word, target_word, show=True):

        candi_phone, target_phone = [i for i in self.separate_to_phone([candidate_word, target_word])]
        diff_info = self.compare_word_phone(candi_phone['phone'], target_phone['phone'])

        def diff_info_to_human():
            human_string_list = []

            def parse_set(phone_set):
                if phone_set == set():
                    return "无"
                s, y, d = '', '', ''
                for i in phone_set:
                    if i[1] == 0:
                        # 声母
                        s = f"{i[0]} "
                    elif i[1] == 1:
                        # 韵母
                        y = f"{i[0]} "
                    else:
                        # 音调
                        d = f"{i[0]}"
                res = ''.join([s, y, d])
                return res

            for item in diff_info:
                here, other, not_exist = item
                string = f"相同: {parse_set(here): <7}| 存在别处:{parse_set(other): <7}|均不存在:{parse_set(not_exist): <7}"
                human_string_list.append(string)
            return human_string_list

        human_str_list = diff_info_to_human()
        if show:
            pprint(human_str_list)
        return human_str_list

    def get_matched_set(self, candidate_word, target_word):
        candi_phone, target_phone = [i for i in self.separate_to_phone([candidate_word, target_word])]
        diff_info = self.compare_word_phone(candi_phone['phone'], target_phone['phone'])
        epd_diff = self.expand_diff_info(diff_info)
        candidate_words = []
        idx = 0
        for i in self.shuffled_all_idiom_array:
            idx += 1
            if self.match(i['phone'], epd_diff):
                candidate_words.append(i['word'])

        return candidate_words

    def calc_word_information(self, word=None, word_phone=None, N=-1):
        if word:
            word_phone = self.separate_to_phone([word])[0]
        all_words = self.shuffled_all_idiom_array
        max_length = len(all_words)
        information_list = []
        for idx, candi_word in enumerate(all_words[:N]):
            diff_info = self.compare_word_phone(candi_list=candi_word['phone'], target_list=word_phone['phone'])
            epd_diff = self.expand_diff_info(diff_info)
            candidate_counts = self.calc_candidate_counts(epd_diff, all_words)
            freq = candidate_counts / max_length
            p = freq
            if p != 0:
                information = - math.log2(p)
            else:
                information = 0
            information_list.append(information)
        return information_list

    @staticmethod
    def info_to_entropy(information_list):
        p_array = np.ones(len(information_list)) / len(information_list)
        entropy = 0
        for p, info in zip(p_array, information_list):
            entropy += p * info
        return entropy

    def calc_words_entropy(self, words_list=None, words_phone_list=None):
        if words_list is not None:
            words_phone_list = self.separate_to_phone(words_list)
        words_entropy_dict = {}
        words_info_dict = {}
        for word_phone in probar(words_phone_list, symbol_1="#"):
            info = self.calc_word_information(word_phone=word_phone, N=1000)
            entropy = self.info_to_entropy(info)
            print(word_phone['word'], ":", entropy)
            words_entropy_dict[word_phone['word']] = entropy
            words_info_dict[word_phone['word']] = info
        return words_entropy_dict, words_info_dict

    def solve(self, candidate_word: str, target_word: str, show_human=True):
        for idx in range(10):
            if show_human:
                self.word_to_human_diff(candidate_word, target_word, show=True)
            candidate_words = self.get_matched_set(candidate_word, target_word)
            candidate_word = candidate_words[np.random.randint(0, len(candidate_words))]
            if candidate_word == target_word:
                print(f"Search {idx + 1} times， find the answer is `{target_word}`. ")
                break

    @staticmethod
    def load_words_entropy_info_dict(multi_num=8, start_idx=0, end_idx=8000):
        step = int((end_idx - start_idx) / multi_num)
        words_entropy_dict = {}
        words_info_dict = {}
        for idx in range(multi_num):
            if idx == multi_num - 1:
                start, end = start_idx + idx * step, end_idx
            else:
                start, end = idx * step, (idx + 1) * step
            entropy_dict = load(rel_to_abs(f"../datasets/words_info/words_entropy_dict_{start}_{end}"))
            info_dict = load(rel_to_abs(f"../datasets/words_info/words_info_dict_{start}_{end}"))
            words_entropy_dict.update(entropy_dict)
            words_info_dict.update(info_dict)
        return words_entropy_dict, words_info_dict


def func(start, end):
    hs = WordleSolver()
    words_entropy_dict, words_info_dict = hs.calc_words_entropy(
        words_phone_list=hs.all_idiom_phone_list[int(start): int(end)])
    save(rel_to_abs(f"../datasets/words_info/words_entropy_dict_{start}_{end}"), words_entropy_dict)
    save(rel_to_abs(f"../datasets/words_info/words_info_dict_{start}_{end}"), words_info_dict)


def multi_run(multi_num=8, start_idx=0, end_idx=8000):
    from multiprocessing import Process
    process_list = []
    step = int((end_idx - start_idx) / multi_num)

    for i in range(multi_num):
        if i == multi_num - 1:
            p = Process(target=func, args=(start_idx + i * step, end_idx))
        else:
            p = Process(target=func, args=(start_idx + i * step, start_idx + (i + 1) * step))
        p.start()
        process_list.append(p)

    [i.join() for i in process_list]


if __name__ == "__main__":
    print("hello world")
    wordler = WordleSolver(use_cache=False)
    print(wordler.idom_content[:10])
    target_word = "虎虎生威"
    human_diff_list = wordler.word_to_human_diff("龙马金属", target_word)
    func(100, 101)

    # mt = MeasureTime()
    # mt.start()
    # multi_run(multi_num=12, start_idx=0, end_idx=1800)
    # mt.show_interval()
