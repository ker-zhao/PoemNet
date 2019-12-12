import json
import os
import h5py
import numpy as np
import random
import collections
# import langconv

cache_dir = "data"
index_file = "data/char_index.json"
data_set_file = "data/dataset.h5"
poem_file = "all_poems.txt"

max_len_poem = 64
max_len_title = 10
max_len_data = max_len_poem + max_len_title + 3
start_char = "S"
end_char = "E"
title_char = "#"
type_5_4 = "A"
type_5_8 = "B"
type_7_4 = "C"
type_7_8 = "D"
type_random = "R"
type_other = "O"
skip_chars = "ABCDRO()[]（）《》_？：" + end_char + title_char + start_char

# def traditional2simplified(sentence):
#     return langconv.Converter('zh-hans').convert(sentence)


def create_poem_json():
    path = "data/poem/"
    data_list = []
    path_list = os.listdir(path)
    for filename in path_list:
        with open(path + filename, "r", encoding='UTF-8') as f:
            data = json.load(f)
            for idx, i in enumerate(data):
                skip = [c for c in skip_chars if c in "".join([i["title"]] + i["paragraphs"])]
                if len(i["paragraphs"]) == 0 or skip or len(i["title"]) > max_len_title:
                    continue
                num_sentence = len(i["paragraphs"])
                len_sentence = len(i["paragraphs"][0].split("，")[0])
                content = "".join(i["paragraphs"])
                if (len_sentence not in [5, 7]) or (num_sentence not in [2, 4]):
                    continue
                if len(content) > max_len_poem or (len(content) % 8 != 0 and len(content) % 6 != 0):
                    continue
                # p = type_char + i["title"] + "#" + content + end_char + "\n"
                p = i["title"] + title_char + content + "\n"
                data_list.append(p)

    with open(poem_file, "w", encoding='UTF-8') as f:
        random.shuffle(data_list)
        f.writelines(data_list)
    with open(poem_file, "r", encoding='UTF-8') as f:
        d = f.read().splitlines()
        print(len(d), d[:10])


def get_type_char(content):
    num_sentence = len(content.strip().split('。')) - 1
    len_sentence = len(content.strip().split('，')[0])

    if random.random() < 0.3:
        type_char = type_random
    else:
        if len_sentence == 5 and num_sentence == 2:
            type_char = type_5_4
        elif len_sentence == 5 and num_sentence == 4:
            type_char = type_5_8
        elif len_sentence == 7 and num_sentence == 2:
            type_char = type_7_4
        elif len_sentence == 7 and num_sentence == 4:
            type_char = type_7_8
        else:
            type_char = type_random
            print("ERROR: unexpected type: ", len_sentence, num_sentence)
    return type_char


def generate_dataset(use_cache=False):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if use_cache and os.path.exists(index_file) and os.path.exists(data_set_file):
        return load_dataset()

    poem_list = []
    with open(poem_file, "r", encoding='UTF-8') as f:
        for line in f.readlines():
            title, content = line.strip().split(title_char)
            title = title.split(' ')[0]
            content = content.replace(' ', '', -1)
            type_char = get_type_char(content)
            # p = start_char + title + title_char + content + end_char
            p = type_char + content + title_char + title + end_char
            poem_list.append(p)
        print(poem_list[:10])
    all_words = [word for poem in poem_list for word in poem]
    counter = collections.Counter(all_words)
    idx2word = sorted(counter.keys(), key=lambda z: counter[z], reverse=True)
    if ' ' not in idx2word:
        idx2word.append(' ')

    word2idx = {v: i for i, v in enumerate(idx2word)}
    indexed_poems = [[word2idx[char] for char in poem] for poem in poem_list]
    space_idx = word2idx[' ']
    x = np.full((len(poem_list), max_len_data), space_idx, np.int32)
    for i, v in enumerate(indexed_poems):
        x[i, :len(v)] = v
    y = np.copy(x)
    y[:, :-1] = x[:, 1:]
    save_dataset(idx2word, word2idx, x, y)
    return idx2word, word2idx, x, y


def save_dataset(idx2word, word2idx, x, y):
    save_dict = {"idx2word": idx2word, "word2idx": word2idx}
    with open(index_file, "w", encoding='UTF-8') as f:
        json.dump(save_dict, f)
    with h5py.File(data_set_file, "w") as f:
        # f = h5py.File(data_set_file, "w")
        f.create_dataset("x", data=x)
        f.create_dataset("y", data=y)
        # f.close()
    return


def load_dataset():
    with open(index_file, "r", encoding='UTF-8') as f:
        idx_dict = json.load(f)
    with h5py.File(data_set_file, 'r') as f:
        return idx_dict["idx2word"], idx_dict["word2idx"], np.array(f["x"]), np.array(f["y"])
    # data = h5py.File(data_set_file, 'r')
    # return idx_dict["idx2word"], idx_dict["word2idx"], np.array(data["x"]), np.array(data["y"])


# def poem_filter():
#     word2vec = {}
#     file = "data/sgns.sikuquanshu.word"
#     with open(file, "r", encoding='UTF-8') as f:
#         for i in f.readlines()[1:]:
#             s = i.split()
#             word2vec[s[0]] = s[1:]
#
#     known_poem = []
#
#     with open("data/poem.json", "r", encoding='UTF-8') as f:
#         data = json.load(f)
#         for i in data:
#             unk = False
#             for j in i:
#                 if j not in word2vec:
#                     unk = True
#
#             if not unk and 0 < len(i) < max_len_poem:
#                 known_poem.append(i)
#     random.shuffle(known_poem)
#     return known_poem, word2vec

#
# def generate_dataset_old():
#     idx_data = get_index()
#     word2index = idx_data["word2index"]
#     word_list = idx_data["word_list"]
#     known_poem, word2vec = poem_filter()
#     known_poem = [i.replace("，", "。") for i in known_poem]
#
#
#     # embedding matrix
#     embed = np.zeros((len(word_list), 300))
#     for (i, v) in enumerate(word_list):
#         embed[i] = np.array(word2vec[v])
#
#     counted_poems = []
#     for i in known_poem:
#         idx = i.find("。")
#         if 0 < idx < 10:
#             counted_poems.append(str(idx) + i)
#
#     # x_test
#     num_test = 1000
#     x_test = np.zeros((num_test, max_len_data), dtype=np.int32)
#     for (idx, poem) in enumerate(counted_poems[:num_test]):
#         poem = poem + "E"
#         poem = np.array([word2index[v] for v in poem])
#         x_test[idx, 0:poem.shape[0]] = poem
#     # x_train
#     x_train = np.zeros((len(counted_poems) - num_test, max_len_data), dtype=np.int32)
#     for (idx, poem) in enumerate(counted_poems[num_test:]):
#         poem = poem + "E"
#         poem = np.array([word2index[v] for v in poem])
#         x_train[idx, 0:poem.shape[0]] = poem
#
#     # y_test
#     y_test = np.zeros((num_test, max_len_data), dtype=np.int32)
#     for (idx, poem) in enumerate(counted_poems[:num_test]):
#         poem = poem[1:] + "E"
#         poem = np.array([word2index[v] for v in poem])
#         y_test[idx, 0:poem.shape[0]] = poem
#     # y_train
#     y_train = np.zeros((len(counted_poems) - num_test, max_len_data), dtype=np.int32)
#     for (idx, poem) in enumerate(counted_poems[num_test:]):
#         poem = poem[1:] + "E"
#         poem = np.array([word2index[v] for v in poem])
#         y_train[idx, 0:poem.shape[0]] = poem
#     print(x_test.shape, x_train.shape, y_test.shape, y_train.shape)
#
#     f = h5py.File(data_set_file, "w")
#     f.create_dataset("embed", data=embed)
#     f.create_dataset("x_test", data=x_test)
#     f.create_dataset("x_train", data=x_train)
#     f.create_dataset("y_test", data=y_test)
#     f.create_dataset("y_train", data=y_train)
#     f.close()
#
#
# def load_dataset():
#     data = h5py.File(data_set_file, 'r')
#     # print(np.array(data["x_test"])[1])
#     # print(np.array(data["y_train"])[1])
#     # print(np.array(data["embed"]).shape, np.array(data["x_test"]).shape, np.array(data["x_train"]).shape,
#     #       np.array(data["y_test"]).shape, np.array(data["y_train"]).shape)
#     return (np.array(data["embed"]), np.array(data["x_test"]), np.array(data["x_train"]),
#             np.array(data["y_test"]), np.array(data["y_train"]))
#
#
# def decode_poem(indexes, index2word):
#     return "".join([index2word[str(i)] for i in indexes])
#
#
# def print_dataset():
#     data = h5py.File(data_set_file, 'r')
#     idx_data = get_index()
#     index2word = idx_data["index2word"]
#     print(decode_poem(np.array(data["x_train"])[10000], index2word))
#     print(decode_poem(np.array(data["y_train"])[10000], index2word))
#
#     # print(np.array(data["x_test"])[1])
#     # print(np.array(data["y_train"])[1])
#     # print(np.array(data["embed"]).shape, np.array(data["x_test"]).shape, np.array(data["x_train"]).shape,
#     #       np.array(data["y_test"]).shape, np.array(data["y_train"]).shape)
#     return
#
#
# def index_chinese():
#     chars = {}
#     known_poem, _ = poem_filter()
#     for i in known_poem:
#         for j in i:
#             chars[j] = 1
#     vob = ["N", "E", "1", "2", "3", "4", "5", "6", "7", "8", "9"] + list(chars.keys())
#
#     word2index = {}
#     index2word = {}
#     for (i, word) in enumerate(vob):
#         word2index[word] = i
#         index2word[i] = word
#     print(len(word2index), word2index["N"], index2word[0])
#     save_dict = {"word2index": word2index, "index2word": index2word, "word_list": vob}
#     with open(char_index_file, "w", encoding='UTF-8') as f:
#         json.dump(save_dict, f)
#
#     # with open(char_index_file, "r", encoding='UTF-8') as f:
#     #     data = json.load(f)
#     # print(data)
#     # data = h5py.File(char_index_file, 'r')
#     # print(data["word2index"])
#
#
# def get_index():
#     with open(char_index_file, "r", encoding='UTF-8') as f:
#         data = json.load(f)
#     return data


def main():
    # create_poem_json()
    generate_dataset(False)
    # index_chinese()
    # generate_dataset()
    # load_dataset()
    # print_dataset()


if __name__ == "__main__":
    main()
