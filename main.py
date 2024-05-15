import tensorflow as tf
import math
import numpy as np
import pandas as pd
from sklearn import model_selection
import re
print(tf.__version__)


data_df = pd.read_csv("data/dialogs_expanded.csv", index_col=False)
data_df = data_df.sample(frac=0.021)
print("Number of elements in features:", data_df.count())
data_df

data_df.drop(['Unnamed: 0','question_as_int','answer_as_int','question_len','answer_len'], axis=1, inplace=True)

def clean_text(text):
  text = text.lower()
  text = re.sub('\[.*?\]', '', text)
  text = re.sub('https?://\S+|www\.\S+', '', text)
  text = re.sub('<.*?>+', '', text)
  text = re.sub('\n', '', text)
  text = re.sub(r'[^\w]',' ',text)
  text = re.sub('\w*\d\w*', '', text)
  #text = re.sub(r'\s+', ' ', text)  # Remove consecutive spaces
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
      filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n', oov_token='<OOV>'
  )
  lang_tokenizer.fit_on_texts(lang)
  tensor = lang_tokenizer.texts_to_sequences(lang)
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
  return tensor, lang_tokenizer

question_sequence, question_tokenizer = tokenize(data_df.question)
answer_sequence, answer_tokenizer = tokenize(data_df.answer)

x_train, x_test, y_train, y_test = model_selection.train_test_split(question_sequence,
                answer_sequence, test_size = 0.1, random_state=42)

x_train.shape, x_test.shape, y_train.shape, y_test.shape

def convert(lang, tensor):
  for t in tensor:
    if t!=0:
      print('%d---> %s' % (t, lang.index_word[t]))

convert(question_tokenizer, x_train[0])
convert(answer_tokenizer, y_train[0])

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

for q, a in train_dataset.take(1):
    print(f'Question:{q.shape}\n{q}')
  
    print(f'Answer:{a.shape}\n{a}')

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

# vocab_inp_size = len(eng_tokenizer.word_index)+1
# vocab_tar_size =  len(spn_tokenizer.word_index)+1
# embedding_dim = 256
# units = 1024
# batch_size=32

encoder = Encoder(vocab_inp_size, embedding_dim, units, batch_size)
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(q, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

decoder = Decoder(vocab_tar_size, embedding_dim, units, batch_size)

sample_decoder_output, _ = decoder(tf.random.uniform((batch_size, 1)), sample_hidden)

print ('Decoder output shape: (batch size, vocab_size) {}'.format(sample_decoder_output.shape))

optimizer = tf.keras.optimizers.Adam()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  return tf.reduce_mean(loss_)

# the training metrics
train_loss = tf.metrics.Mean(name='train loss')
# training loss
test_loss =tf.metrics.Mean(name='test loss')

accuracy = tf.keras.metrics.Accuracy()

# create the training step
# using the tf.function decorator to speed up the training process by converting the training function to a TensorFlow graph
@tf.function
def train_step(inputs, target, encoder_hidden_state):
    total_loss = 0

    with tf.GradientTape() as tape:
        encoder_output, encoder_hidden_state = encoder(inputs, encoder_hidden_state)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = tf.expand_dims([answer_tokenizer.word_index['<start>']] * inputs.shape[0], 1)

        for timestep in range(1, target.shape[1]):
            predictions, decoder_hidden_state = decoder(decoder_input, decoder_hidden_state)
            total_loss += loss_function(target[:, timestep], predictions)
            decoder_input = tf.expand_dims(target[:, timestep], 1)

    batch_loss = (total_loss / int(target.shape[1]))

    all_trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(total_loss, all_trainable_variables)
    optimizer.apply_gradients(zip(gradients, all_trainable_variables))
    train_loss(batch_loss)

    return batch_loss

@tf.function
def test_step(inputs, target, enc_hidden_state):
    loss = 0

    enc_output, enc_hidden_state = encoder(inputs, enc_hidden_state)
    dec_hidden_state = enc_hidden_state
    dec_input = tf.expand_dims([answer_tokenizer.word_index['<start>']] * inputs.shape[0], 1)
    
    for time_step in range(1, target.shape[1]):
        predictions, dec_hidden_state = decoder(dec_input, dec_hidden_state)
        loss += loss_function(target[:, time_step], predictions)
        dec_input = tf.expand_dims(target[:, time_step], 1)
        
    batch_loss = (loss / int(target.shape[1]))
    test_loss(batch_loss)
    return batch_loss

# Initialize lists to store metrics
train_losses = []
test_losses = []

EPOCHS = 10
old_test_loss = math.inf # extremly high number here.

for epoch in range(EPOCHS):
    train_loss.reset_state()
    test_loss.reset_state()

    enc_hidden = encoder.initialize_hidden_state()
    steps_per_epoch = answer_sequence.shape[0] // batch_size
    bar = tf.keras.utils.Progbar(target=steps_per_epoch)

    count = 0

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
        encoder.save(filepath='encoder.keras')
        decoder.save(filepath='decoder.keras')
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

"""
while True:
    user_input = input("You: ")
    if user_input.lower() == 'q':
        print("Goodbye!")
        break
    else:
        _, response = generate_response(user_input)
        print("Bot:", response)
"""