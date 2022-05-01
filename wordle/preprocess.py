from sparrow import load, save, rel_to_abs
import pandas as pd
from wordfreq import zipf_frequency
from pypinyin.contrib.tone_convert import to_tone3, to_tone2


def get_high_freq_words(filename=rel_to_abs('../datasets/idiom.json'),
                        save_name=rel_to_abs("../datasets/words_freq.csv")):
    df = pd.read_json(filename)
    word_list, freq_list, pinyin_list = [], [], []
    for word, pinyin in zip(df['word'], df['pinyin']):
        if len(word) == 4:
            word_list.append(word)
            freq_list.append(zipf_frequency(word, 'zh'))
            pinyin_list.append(to_tone2(pinyin))

    df = pd.DataFrame({'word': word_list,  'pinyin': pinyin_list, 'freq': freq_list})
    df = df.sort_values(by='freq', ascending=False)
    df.to_csv(save_name, index=False)

if __name__ == "__main__":
    # print(get_four_words_idiom(use_cache=False, save_to_csv=True))
    # print(get_four_words_idiom(use_cache=True))
    get_high_freq_words()
    # df = pd.read_csv(rel_to_abs("../datasets/words_freq.csv"))
    # df.plot()

