from train_functions import *
from tensorflow.keras.models import load_model

# Load the saved models
encoder = load_model('/content/models/encoder')
decoder = load_model('/content/models/decoder')

def generate_response(sentence):
    cleaned_sentence = clean_text(sentence)
    preprocessed_sentence = add_start_end(cleaned_sentence)
    inputs = question_tokenizer.texts_to_sequences([preprocessed_sentence])
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=29, padding='post')

    initial_hidden_state = [tf.zeros((1, units))]
    encoder_output, encoder_hidden_state = encoder(inputs, initial_hidden_state)
    decoder_hidden_state = encoder_hidden_state
    decoder_input = tf.expand_dims([answer_tokenizer.word_index['<start>']], 0)
    result = ''

    for _ in range(32):
        predictions, decoder_hidden_state = decoder(decoder_input, decoder_hidden_state)
        predicted_id = tf.argmax(predictions[0]).numpy()
        result += answer_tokenizer.index_word[predicted_id] + ' '
        if answer_tokenizer.index_word[predicted_id] == '<end>':
            result = result.replace('<start> ', '')
            result = result.replace(' <end> ', '')
            cleaned_sentence = cleaned_sentence.replace('<start> ', '')
            cleaned_sentence = cleaned_sentence.replace(' <end>', '')
            return cleaned_sentence, result

        decoder_input = tf.expand_dims([predicted_id], 0)

    result = result.replace('<start> ', '')
    result = result.replace('<end>', '')
    cleaned_sentence = cleaned_sentence.replace('<start> ', '')
    cleaned_sentence = cleaned_sentence.replace('<end>', '')

    return cleaned_sentence, result



print(generate_response("how are you today"))