import gradio as gr
from wordle import WordleSolver
import numpy as np

wordle_solver = WordleSolver(use_cache=True)
target_word = np.random.choice(wordle_solver.shuffled_all_idiom_array)['word']
words_entropy_dict, words_info_dict = wordle_solver.load_words_entropy_info_dict()


def func(word: str, update: bool):
    global target_word
    if update:
        target_word = np.random.choice(wordle_solver.shuffled_all_idiom_array)['word']
    human_diff_list = wordle_solver.word_to_human_diff(word, target_word)
    return human_diff_list


iface = gr.Interface(fn=func,
                     inputs=[gr.inputs.Textbox(lines=2, placeholder="请四字成语..."),
                             'checkbox'],
                     outputs="text",
                     title="中文wordle",
                     # description=description,
                     # article=article,
                     examples=[["龙争虎斗", False]]
                     )

iface.launch(
    debug=True,
    server_name='0.0.0.0',
    server_port=57860
)
