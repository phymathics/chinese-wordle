import streamlit as st
import sys

sys.path.append('..')
from wordle import WordleSolver, color_str
import numpy as np
from sparrow.performance import MeasureTime
from copy import copy


class NewWordle(WordleSolver):
    def word_to_human_diff(self, candidate_word, target_word, show=True):

        candi_phone, target_phone = [i for i in self.separate_to_phone([candidate_word, target_word])]
        diff_info = self.compare_word_phone(candi_phone['phone'], target_phone['phone'])

        def get_color_str(text, idx):
            if idx == 0:  # here
                return color_str(text, )  # green
            elif idx == 1:
                return color_str(text, "#FF8000")  # yellow
            else:
                return color_str(text, "#616161")  # grey

        def parse_item(item):
            s, y, d = '', '', ''
            for idx, set_i in enumerate(item):
                if set_i == set():
                    continue
                for phone_tuple in set_i:
                    if phone_tuple[1] == 0:
                        s = get_color_str(phone_tuple[0], idx)
                    elif phone_tuple[1] == 1:
                        y = get_color_str(phone_tuple[0], idx)
                    elif phone_tuple[1] == 2:
                        d = get_color_str(phone_tuple[0], idx)
                    else:
                        raise ValueError
            return f"{s}{y}{d}"

        human_string_combined = ""
        for item in diff_info:
            human_string_combined += "&ensp;" + parse_item(item)

        return human_string_combined


def parse_chinese_diff(words_origin: str, target_words_origin: str):
    words, target_words = [i for i in words_origin], [i for i in target_words_origin]
    idx_dict = {
        'here': [],
        'other': [],
        'not_exsit': []
    }
    for idx in range(len(words) - 1, -1, -1):
        word, target_word = words[idx], target_words[idx]
        if word == target_word:
            idx_dict['here'].append(idx)
            words.pop(idx)
            target_words.pop(idx)

    for idx in range(len(words) - 1, -1, -1):
        for jdx in range(len(target_words) - 1, -1, -1):
            word, target_word = words[idx], target_words[jdx]
            if word == target_word:
                idx_dict['other'].append(idx)
                words.pop(idx)
                target_words.pop(jdx)
                break
        idx_dict['not_exsit'].append(idx)

    def coloring_words():
        colored_words = ""
        for idx, word in enumerate(words_origin):
            if idx in idx_dict['here']:
                colored_words += " " + color_str(word)  # green
            elif idx in idx_dict['other']:
                colored_words += " " + color_str(word, '#FF8000')  # yellow
            else:
                colored_words += " " + color_str(word, '#616161')  # grey
        return colored_words

    return coloring_words()


# wordle_solver, target_word = init()
# words_entropy_dict, words_info_dict = wordle_solver.load_words_entropy_info_dict()


if 'wordler' not in st.session_state:
    st.session_state.setdefault('wordler', NewWordle(use_cache=True))
    st.session_state.setdefault('target_word',
                                np.random.choice(st.session_state['wordler'].shuffled_all_idiom_array)['word'])

clear_texts = False
if st.button("更新成语"):
    st.session_state['target_word'] = np.random.choice(st.session_state['wordler'].shuffled_all_idiom_array)['word']
    clear_texts = True

look_over_expander = st.expander("查看成语", expanded=False)
look_over_expander.write(st.session_state['target_word'])

word = st.text_input("情输入四字词语", "龙争虎斗")
human_diff = st.session_state['wordler'].word_to_human_diff(word, st.session_state['target_word'])
chinese_diff = parse_chinese_diff(word, st.session_state['target_word'])
st.session_state.setdefault('word', [])
st.session_state['word'].append((human_diff, chinese_diff))

if clear_texts or st.button("清除历史"):
    st.session_state['word'] = []

for text in st.session_state['word']:
    st.markdown(text[0], unsafe_allow_html=True)
    st.markdown(text[1], unsafe_allow_html=True)
