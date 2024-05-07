import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from data_preprocessing import *
import re

# Define your data preprocessing functions, such as clean_text, add_start_end, and tokenize

# Define your chatbot function
def chatbot(sentence):
    # Load the tokenizer and model
    question_tokenizer = tf.keras.preprocessing.text.Tokenizer()
    answer_tokenizer = tf.keras.preprocessing.text.Tokenizer()
    question_tokenizer.fit_on_texts([''])
    answer_tokenizer.fit_on_texts([''])

    encoder = load_model('encoder_model.h5')
    decoder = load_model('decoder_model.h5')

    # Preprocess the input sentence
    sentence = clean_text(sentence)
    sentence = add_start_end(sentence)
    inputs = question_tokenizer.texts_to_sequences([sentence])
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=29, padding='post')

    # Initialize hidden state for encoder
    hidden = [tf.zeros((1, 1024))]

    # Get encoder output and hidden state
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([answer_tokenizer.word_index['<start>']], 0)
    result = ''

    # Generate the response
    for t in range(32):
        predictions, dec_hidden = decoder(dec_input, dec_hidden)
        predicted_id = tf.argmax(predictions[0]).numpy()
        result += answer_tokenizer.index_word[predicted_id] + ' '
        if answer_tokenizer.index_word[predicted_id] == '<end>':
            result = result.replace('<start> ', '')
            result = result.replace(' <end> ', '')
            sentence = sentence.replace('<start> ', '')
            sentence = sentence.replace(' <end>', '')
            return sentence, result

        dec_input = tf.expand_dims([predicted_id], 0)

    result = result.replace('<start> ', '')
    result = result.replace('<end>', '')
    sentence = sentence.replace('<start> ', '')
    sentence = sentence.replace('<end>', '')

    return sentence, result

# Test the chatbot function
print(chatbot("how are you today"))
print(chatbot('what is the weather outside'))
print(chatbot(' what is the weather outside'))
print(chatbot(' how old '))
print(chatbot('can you play'))
