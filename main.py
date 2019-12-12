from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import os

import utils


model_file = 'DeepPoemModel.h5'
checkpoint_dir = './training_checkpoints'

batch_size = 64
epochs = 50
learning_rate = 0.001
drop_rate = 0.05
word_vec_size = 256
rnn_size = 1024

idx2word, word2idx, x_train, y_train = utils.generate_dataset(True)
vob_size = len(idx2word)


def create_model_cell():
    input_data = layers.Input(shape=(None,))
    initial_state_1 = [layers.Input(shape=(128,)), layers.Input(shape=(128,))]
    initial_state_2 = [layers.Input(shape=(128,)), layers.Input(shape=(128,))]
    x = layers.Embedding(input_dim=vob_size, output_dim=word_vec_size)(input_data)

    x, state_h_1, state_c_1 = layers.LSTM(
        128, return_state=True, return_sequences=True)(x, initial_state=initial_state_1)
    state_1 = [state_h_1, state_h_1]
    x, state_h_2, state_c_2 = layers.LSTM(
        128, return_state=True, return_sequences=True)(x, initial_state=initial_state_2)
    state_2 = [state_h_2, state_h_2]
    x = layers.Dense(vob_size)(x)

    model = tf.keras.Model([input_data, initial_state_1, initial_state_2], [x, state_1, state_2])
    model.summary()


def create_model(is_sampling=False):
    _batch_size = 1 if is_sampling else batch_size
    model = tf.keras.Sequential()
    model.add(layers.Embedding(input_dim=vob_size, output_dim=word_vec_size,
                               batch_input_shape=[_batch_size, None]))
    model.add(layers.LSTM(rnn_size, return_sequences=True, stateful=is_sampling))
    # model.add(layers.Dropout(drop_rate))
    model.add(layers.LSTM(rnn_size, return_sequences=True, stateful=is_sampling))
    # model.add(layers.Dropout(drop_rate))
    # model.add(layers.LSTM(rnn_size, return_sequences=True, stateful=is_sampling))
    # model.add(layers.Dropout(drop_rate))
    # model.add(layers.Dense(rnn_size, activation="relu"))
    # model.add(layers.Dropout(drop_rate))

    model.add(layers.Dense(vob_size))
    return model


def create_train():
    model = create_model()
    train_model(model)


def train():
    model = create_model()
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    train_model(model)


def train_model(model):
    model.summary()
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    model.compile(tf.optimizers.Adam(learning_rate=learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint_callback])


def generate_text(model, type_str="R", start_string="", temperature=0.7, acrostic_str=""):
    num_generate = 100
    start_string = type_str + start_string
    # 将起始字符串转换为数字（向量化）
    input_eval = [word2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []

    ac_idx = 0
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # 删除批次的维度
        predictions = tf.squeeze(predictions, 0)
        # 用分类分布预测模型返回的字符
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        # predictions = tf.nn.softmax(predictions)
        # char = to_word(predictions, idx2word)
        # predicted_id = word2idx[char]
        if idx2word[predicted_id] == utils.end_char:
            break
        if acrostic_str:
            if (i == 0 or text_generated[-1] == "，" or text_generated[-1] == "。") and ac_idx < len(acrostic_str):
                predicted_id = word2idx[acrostic_str[ac_idx]]
                ac_idx += 1

        # 把预测字符和前面的隐藏状态一起传递给模型作为下一个输入
        text_generated.append(idx2word[predicted_id])
        input_eval = tf.expand_dims([predicted_id], 0)
        # input_eval = [word2idx[s] for s in start_string + ''.join(text_generated)]
        # input_eval = tf.expand_dims(input_eval, 0)
    return start_string[1:] + ''.join(text_generated)


def sampling():
    model = create_model(True)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    model.build(tf.TensorShape([1, None]))
    model.summary()
    temperature = [0.01, 0.03, 0.1, 0.3, 1, 3]
    temperature2 = [0.1 * i + 0.3 for i in range(10)]
    temperature = temperature + temperature2
    for i in temperature:
        print(i, "\t\t: ", generate_text(model, "C", "", i))


def compose():
    model = create_model(True)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    model.build(tf.TensorShape([1, None]))
    temperature = [0.1 * i + 0.1 for i in range(10)]
    for i in temperature:
        print(format_poem(generate_text(model, "C", "", i, "")))


def format_poem(poem):
    return "。\n".join(poem.split("。")[:-1]) + "。\n#" + poem.split("#")[1] + "\n"


def main():
    # create_train()
    compose()
    # train()
    # sampling()


if __name__ == "__main__":
    main()
