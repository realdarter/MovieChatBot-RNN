from train_functions import *
import math
import tensorflow as tf
import os
from encoder import Encoder
from decoder import Decoder

# Check if the directory exists, if not, create it
directory = '/content/models'
if not os.path.exists(directory):
    os.makedirs(directory)

# Initialize lists to store metrics
train_losses = []
test_losses = []

EPOCHS = 10
old_test_loss = math.inf  # extremely high number here.

for epoch in range(EPOCHS):
    train_loss.reset_state()
    test_loss.reset_state()

    enc_hidden = encoder.initialize_hidden_state()
    steps_per_epoch = len(train_dataset)
    bar = tf.keras.utils.Progbar(target=steps_per_epoch)

    for step, (inputs, target) in enumerate(train_dataset):
        batch_loss = train_step(inputs, target, enc_hidden)
        bar.update(step + 1)

    for step, (inputs, target) in enumerate(test_dataset):
        test_step(inputs, target, enc_hidden)

    if old_test_loss > test_loss.result():
        old_test_loss = test_loss.result()
        encoder.save(filepath='/content/models/encoder.keras')  # Save encoder model with .keras extension
        decoder.save(filepath='/content/models/decoder.keras')  # Save decoder model with .keras extension
        print('Model is saved')

    train_losses.append(train_loss.result())
    test_losses.append(test_loss.result())
    
    print('#' * 50)
    print(f'Epoch #{epoch + 1}')
    print(f'Training Loss {train_loss.result()}')
    print(f'Testing Loss {test_loss.result()}')
    print('#' * 50)

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
      result = result.replace(' <end> ','')
      cleaned_sentence = cleaned_sentence.replace('<start> ', '')
      cleaned_sentence = cleaned_sentence.replace(' <end>', '')
      return  cleaned_sentence, result

    decoder_input = tf.expand_dims([predicted_id], 0)

  result = result.replace('<start> ', '')
  result = result.replace('<end>','')
  cleaned_sentence = cleaned_sentence.replace('<start> ', '')
  cleaned_sentence = cleaned_sentence.replace('<end>', '')

  return cleaned_sentence, result

print(generate_response("how are you today"))

while True:
    user_input = input("You: ")
    if user_input.lower() == 'q':
        print("Goodbye!")
        break
    else:
        _, response = generate_response(user_input)
        print("Bot:", response)

