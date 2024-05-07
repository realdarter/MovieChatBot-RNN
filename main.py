import tensorflow as tf

def chatbot(sentence, encoder, decoder, question_tokenizer, answer_tokenizer, units):
    sentence = clean_text(sentence)
    sentence = add_start_end(sentence)
    inputs = question_tokenizer.texts_to_sequences([sentence])
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=29, padding='post')
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([answer_tokenizer.word_index['<start>']], 0)
    result = ''
    for t in range(32):
        predictions, dec_hidden = decoder(dec_input, dec_hidden)
        predicted_id = tf.argmax(predictions[0]).numpy()
        result += answer_tokenizer.index_word[predicted_id] + ' '
        if answer_tokenizer.index_word[predicted_id] == '<end>':
            result = result.replace('<start> ', '')
            result = result.replace(' <end> ','')
            sentence = sentence.replace('<start> ', '')
            sentence = sentence.replace(' <end>', '')
            return  sentence, result
        dec_input = tf.expand_dims([predicted_id], 0)
    result = result.replace('<start> ', '')
    result = result.replace('<end>','')
    sentence = sentence.replace('<start> ', '')
    sentence = sentence.replace('<end>', '')
    return sentence, result
