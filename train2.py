import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import model_selection
import re

data_df = pd.read_csv('data/dialogs_expanded.csv', index_col=False)

data_df = data_df.sample(frac=0.01, random_state=42)

data_df.drop(['Unnamed: 0','question_as_int','answer_as_int','question_len','answer_len'], axis=1, inplace=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

data_df.question = data_df.question.map(clean_text)
data_df.answer = data_df.answer.map(clean_text)

def add_start_end(text):
  text = f'<start> {text} <end>'
  return text

data_df.question = data_df.question.map(add_start_end)
data_df.answer = data_df.answer.map(add_start_end)

def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n',
        oov_token='<OOV>'
    )
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer

question_sequence, question_tokenizer = tokenize(data_df.question)
answer_sequence, answer_tokenizer = tokenize(data_df.answer)

x_train, x_test, y_train, y_test = model_selection.train_test_split(question_sequence, answer_sequence, test_size = 0.1, random_state=42) 

vocab_inp_size = len(question_tokenizer.word_index)+1
vocab_tar_size =  len(answer_tokenizer.word_index)+1
embedding_dim = 256
units = 1024
batch_size=32

def create_dataset(x, y, batch_size=32):
  data = tf.data.Dataset.from_tensor_slices((x, y))
  data = data.shuffle(1028)
  data = data.batch(batch_size, drop_remainder=True)
  data = data.prefetch(tf.data.experimental.AUTOTUNE)
  return data

train_dataset = create_dataset(x_train, y_train)
test_dataset = create_dataset(x_test, y_test)

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, encoder_units, batch_size):
      super(Encoder, self).__init__()

      self.batch_size = batch_size
      self.encoder_units = encoder_units
      self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
      self.gru = tf.keras.layers.GRU(self.encoder_units, 
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer = 'glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_size, self.encoder_units))
  
class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, decoder_units, batch_size):
      super(Decoder, self).__init__()

      self.batch_size = batch_size
      self.decoder_units = decoder_units
      self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
      self.gru = tf.keras.layers.GRU(self.decoder_units, 
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer = 'glorot_uniform')
      
      self.fc = tf.keras.layers.Dense(vocab_size)


  def call(self, x, hidden):
    x = self.embedding(x)
    output, hidden = self.gru(x, initial_state = hidden)
    output = tf.reshape(output, (-1, output.shape[2]))
    x =  tf.nn.softmax(self.fc(output))
    return x, hidden
  
encoder = Encoder(vocab_inp_size, embedding_dim, units, batch_size)
decoder = Decoder(vocab_tar_size, embedding_dim, units, batch_size)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  return tf.reduce_mean(loss_)

train_loss = tf.metrics.Mean(name='train loss')
test_loss =tf.metrics.Mean(name='test loss')

@tf.function
def train_step(inputs, target, enc_hidden):
  loss = 0
  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inputs, enc_hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([answer_tokenizer.word_index['<start>']] * inputs.shape[0], 1)
    for t in range(1, target.shape[1]):
      predictions, dec_hidden = decoder(dec_input, dec_hidden)
      loss += loss_function(target[:, t], predictions)
      dec_input = tf.expand_dims(target[:, t], 1)
  batch_loss = (loss / int(target.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))
  train_loss(batch_loss)
  return batch_loss

@tf.function 
def test_step(inputs, target, enc_hidden):
    loss = 0
    enc_output, enc_hidden = encoder(inputs, enc_hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([answer_tokenizer.word_index['<start>']] * inputs.shape[0], 1)
    for t in range(1, target.shape[1]):
        predictions, dec_hidden = decoder(dec_input, dec_hidden)
        loss += loss_function(target[:, t], predictions)
        dec_input = tf.expand_dims(target[:, t], 1)
    batch_loss = (loss / int(target.shape[1]))
    test_loss(batch_loss)

EPOCHS = 2
old_test_loss=1000000

for epoch in range(EPOCHS):
    train_loss.reset_state()
    test_loss.reset_state()
    enc_hidden = encoder.initialize_hidden_state()
    steps_per_epoch = answer_sequence.shape[0]//batch_size 
    bar = tf.keras.utils.Progbar(target=steps_per_epoch)
    
    count=0
    for (batch, (inputs, target)) in enumerate(train_dataset):
        count += 1
        batch_loss = train_step(inputs, target, enc_hidden)
        bar.update(count)
                                                  
    for (batch, (inputs, target)) in enumerate(test_dataset):
        count += 1
        batch_loss = test_step(inputs, target, enc_hidden)
    bar.update(count)
    
    if old_test_loss > test_loss.result():
        old_test_loss = test_loss.result()
        encoder.save(filepath='encoder_model.h5')
        decoder.save(filepath='decoder_model.h5')
        print('Model is saved')
    
    print('#' * 50)
    print(f'Epoch #{epoch + 1}')
    print(f'Training Loss {train_loss.result()}')
    print(f'Testing Loss {test_loss.result()}')
    print('#' * 50)

def chatbot(sentence):
  sentence = clean_text(sentence)
  sentence = add_start_end(sentence)
  inputs = question_tokenizer.texts_to_sequences([sentence])
  inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                         maxlen=29,
                                                         padding='post')
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

print(chatbot("how are you today"))
print(chatbot('what is the weather outside'))
print(chatbot(' what is the weather outside'))
print(chatbot(' how old '))
print(chatbot('can you play'))
