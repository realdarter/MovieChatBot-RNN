import pandas as pd
import numpy as np
from sklearn import model_selection
import re
import tensorflow as tf

def read_data(file_path):
    data_df = pd.read_csv(file_path, index_col=False)
    return data_df

def sample_data(data_df, fraction=0.01, seed=42):
    return data_df.sample(frac=fraction, random_state=seed)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def add_start_end(text):
    return f'<start> {text} <end>'

def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n',
        oov_token='<OOV>'
    )
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer

def preprocess_data(data_df):
    data_df.drop(['Unnamed: 0','question_as_int','answer_as_int','question_len','answer_len'], axis=1, inplace=True)
    data_df['question'] = data_df['question'].map(clean_text)
    data_df['answer'] = data_df['answer'].map(clean_text)
    data_df['question'] = data_df['question'].map(add_start_end)
    data_df['answer'] = data_df['answer'].map(add_start_end)
    return data_df

def create_dataset(inputs, targets, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    dataset = dataset.shuffle(len(inputs))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def split_dataset(question_sequence, answer_sequence):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(question_sequence, answer_sequence, test_size=0.1, random_state=42)
    return x_train, x_test, y_train, y_test
