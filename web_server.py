from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional, List, Union, Dict
from wordle import WordleSolver
import numpy as np
import uvicorn

app = FastAPI()
wordle_solver = WordleSolver(use_cache=True)
target_word = np.random.choice(wordle_solver.shuffled_all_idiom_array)['word']
words_entropy_dict, words_info_dict = wordle_solver.load_words_entropy_info_dict()


class WordleModel(BaseModel):
    word: str = Field(..., description="输入四字词语")
    update: bool = Field(False, description="是否重新生成目标词语")


class WordInfo(BaseModel):
    word: str = Field(..., description="输入四字成语")


@app.post("/wordle")
async def post_wordle(data: WordleModel):
    global target_word
    word = data.word
    if data.update:
        target_word = np.random.choice(wordle_solver.shuffled_all_idiom_array)['word']
    print(f"candidate: {word}  target_word: {target_word}")
    human_diff_list = wordle_solver.word_to_human_diff(word, target_word)
    return human_diff_list


@app.post("/word_info")
def get_word_entropy_info(data: WordInfo):
    entropy = words_entropy_dict.get(data.word)
    information = words_info_dict.get(data.word)
    if entropy is None:
        return "未找到该词"
    return {
        "entropy": entropy,
        "information": information
    }


if __name__ == "__main__":
    uvicorn.run(
        app="web_server:app",
        host='0.0.0.0',
        port=9527
    )
