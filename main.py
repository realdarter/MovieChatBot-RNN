import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import model_selection
import re
print(tf.config.list_physical_devices('GPU'))

data_df = pd.read_csv('data/dialogs_expanded.csv', index_col=False)

data_df.drop(['Unnamed: 0','question_as_int','answer_as_int','question_len','answer_len'], axis=1, inplace=True)

data_df.info()

data_df

# Define a function to clean text data
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)  # Remove anything within square brackets
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'\n', '', text)  # Remove newline characters
    text = re.sub(r'[^\w\s]', '', text)  # Remove non-alphanumeric characters (excluding spaces)
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing digits
    return text

data_df.question = data_df.question.map(clean_text)
data_df.answer = data_df.answer.map(clean_text)

def add_start_end(text):
  text = f'<start> {text} <end>'
  return text

data_df.question = data_df.question.map(add_start_end)
data_df.answer = data_df.answer.map(add_start_end)

def tokenize(lang):
    # Initialize a Tokenizer for the language
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n', # Define characters to filter out
        oov_token='<OOV>'  # Token for out-of-vocabulary words
    )
    # Fit the tokenizer on the language data
    lang_tokenizer.fit_on_texts(lang)
    # Convert text sequences to sequences of token indices
    tensor = lang_tokenizer.texts_to_sequences(lang)
    # Pad sequences to ensure uniform length
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer

# Tokenize questions and answers
question_sequence, question_tokenizer = tokenize(data_df.question)
answer_sequence, answer_tokenizer = tokenize(data_df.answer)

x_train, x_test, y_train, y_test = model_selection.train_test_split(question_sequence, answer_sequence, test_size = 0.1, random_state=42) 

x_train.shape, x_test.shape, y_train.shape, y_test.shape


def convert(lang, tensor):
  for t in tensor:
    if t!=0:
      print('%d---> %s' % (t, lang.index_word[t]))

print('Question')
convert(question_tokenizer, x_train[0])
print()
print('Answer')
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

# create the optimizer using the Adam optimizer
optimizer = tf.keras.optimizers.Adam()
# create the loss function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction='none')

# define the loss function for the training
def loss_function(real, pred):
  # create the mask to ignore the padding tokens
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  # mask shape == (batch_size, sequence_length)
  # calculate the loss
  loss_ = loss_object(real, pred)
  # mask the loss
  # how the mask works:
  # if the value is 1, the loss is calculated
  # if the value is 0, the loss is ignored
    #[1,1,1,1,1,1,0,0,0,0,0] mask
    # *
    #[2,6,2,1,6,3,2,1,5,7,9] input
    # =
    #[2,6,2,1,6,3,0,0,0,0,0] output
  mask = tf.cast(mask, dtype=loss_.dtype)
  # mask shape == (batch_size, sequence_length)

  loss_ *= mask
  # calculate the average loss per batch 
  return tf.reduce_mean(loss_)

# create the training metric 
train_loss = tf.metrics.Mean(name='train loss')
# create the testing metric 
test_loss =tf.metrics.Mean(name='test loss')

# create the training step
# using the tf.function decorator to speed up the training process by converting the training function to a TensorFlow graph
@tf.function
# define the training step 
def train_step(inputs, target, enc_hidden):
  # the encoder_hidden is the initial hidden state of the encoder
  # enc_hidden shape == (batch_size, hidden_size)

  # inilaize the loss to zero
  loss = 0
  # create the gradient tape to record the gradient of the loss with respect to the weights

  with tf.GradientTape() as tape:
    # pass the input to the encoder
    # enc_output shape == (batch_size, 49, hidden_size)
    # enc_hidden shape == (batch_size, hidden_size)
    # using the encoder to get the encoder_output and the encoder_hidden
    # using the encoder_hidden as the initial hidden state of the decoder
    enc_output, enc_hidden = encoder(inputs, enc_hidden)
    # set the initial decoder hidden state to the encoder hidden state
    dec_hidden = enc_hidden

    # create the start token 
    # start_token shape == (batch_size, 1)
    # repeat the start token for the batch size times
    dec_input = tf.expand_dims([answer_tokenizer.word_index['<start>']] * inputs.shape[0], 1)
    
    # Teacher forcing - feeding the target as the next input
    
    for t in range(1, target.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden = decoder(dec_input, dec_hidden)
      # calculate the loss for the current time step using the loss function
      loss += loss_function(target[:, t], predictions)

      # using teacher forcing
      dec_input = tf.expand_dims(target[:, t], 1)
  # calculate the loss for the current batch
  batch_loss = (loss / int(target.shape[1]))

  # get the trainable variables
  variables = encoder.trainable_variables + decoder.trainable_variables
  # calculate the gradients using the tape 
  gradients = tape.gradient(loss, variables)
  # update the trainable variables
  optimizer.apply_gradients(zip(gradients, variables))
  # add the loss to the training loss metric
  train_loss(batch_loss)
  return batch_loss

# create the training step
# using the tf.function decorator to speed up the training process by converting the training function to a TensorFlow graph
@tf.function 
def test_step(inputs, target, enc_hidden):
    # the encoder_hidden is the initial hidden state of the encoder
    # enc_hidden shape == (batch_size, hidden_size)
    # inilaize the loss to zero
    loss = 0
    # pass the input to the encoder 
    # enc_output shape == (batch_size, 49, hidden_size) 
    # enc_hidden shape == (batch_size, hidden_size)
    # using the encoder to get the encoder_output and the encoder_hidden
    enc_output, enc_hidden = encoder(inputs, enc_hidden)
    # set the initial decoder hidden state to the encoder hidden state
    dec_hidden = enc_hidden
    # create the start token
    # start_token shape == (batch_size, 1)
    # repeat the start token for the batch size times
    dec_input = tf.expand_dims([answer_tokenizer.word_index['<start>']] * inputs.shape[0], 1)
    for t in range(1, target.shape[1]):
        # passing enc_output to the decoder with dec_hidden as the initial hidden state
        predictions, dec_hidden = decoder(dec_input, dec_hidden)
        # calculate the loss for the current time step using the loss function 
        loss += loss_function(target[:, t], predictions)

        # using teacher forcing
        dec_input = tf.expand_dims(target[:, t], 1)
    # calculate the loss for the current batch
    batch_loss = (loss / int(target.shape[1]))
    # add the batch loss to the test loss metric
    test_loss(batch_loss)

    # set the epochs to 10
EPOCHS = 5
# set the old test loss to high number 

old_test_loss=1000000
# create the training loop
# Assuming you have already imported necessary libraries and defined functions

# Select a subset of the data
small_data_df = data_df.head(300)

# Split the subset into training and testing sets
small_x_train, small_x_test, small_y_train, small_y_test = model_selection.train_test_split(
    question_sequence[:1000], answer_sequence[:1000], test_size=0.1, random_state=42)

# Create datasets for training and testing
small_train_dataset = create_dataset(small_x_train, small_y_train)
small_test_dataset = create_dataset(small_x_test, small_y_test)

# Train the model on the smaller dataset
for epoch in range(EPOCHS):
    train_loss.reset_state()
    test_loss.reset_state()

    enc_hidden = encoder.initialize_hidden_state()
    steps_per_epoch = small_y_train.shape[0] // batch_size
    bar = tf.keras.utils.Progbar(target=steps_per_epoch)

    for (batch, (inputs, target)) in enumerate(small_train_dataset):
        batch_loss = train_step(inputs, target, enc_hidden)
        bar.update(batch + 1)  

    for (batch, (inputs, target)) in enumerate(small_test_dataset):
        batch_loss = test_step(inputs, target, enc_hidden)
        bar.update(batch + 1)

    # Print training and testing loss
    print('#' * 50)
    print(f'Epoch #{epoch + 1}')
    print(f'Training Loss {train_loss.result()}')
    print(f'Testing Loss {test_loss.result()}')
    print('#' * 50)


    # create the chatbot function
# the chatbot function takes in the question as input and answers the input sentence 
def chatbot(sentence):
  
  # clean the input question sentence 
  sentence = clean_text(sentence)
  # add the start token to the sentence
  sentence =add_start_end(sentence)
  # tokenize the sentence
  inputs = question_tokenizer.texts_to_sequences([sentence])
  # pad the sentence
  inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                         maxlen=29,
                                                         padding='post')
  
  # initalize the hidden state of the encoder to zeros
  hidden = [tf.zeros((1, units))]
  # pass the sentence to the encoder with the hidden state as the initial hidden state
  enc_out, enc_hidden = encoder(inputs, hidden)
  # set the initial decoder hidden state to the encoder hidden state
  dec_hidden = enc_hidden
  # create the start token
  # start_token shape == (batch_size, 1)
  # repeat the start token for the batch size times
  dec_input = tf.expand_dims([answer_tokenizer.word_index['<start>']], 0)
  # create the result string
  result = ''
  # loop over the length of the sentence (32)

  for t in range(32):
    # passing the encoder output and the decoder hidden state to the decoder make sure the decoder input is the previous predicted word
    predictions, dec_hidden = decoder(dec_input, dec_hidden)

    # getting the predicted word index
    predicted_id = tf.argmax(predictions[0]).numpy()
    # getting the predicted word using the predicted index
    # add the predicted word to the result string 
    result += answer_tokenizer.index_word[predicted_id] + ' '
    # if the predicted word is the <end> token then stop the loop
    if answer_tokenizer.index_word[predicted_id] == '<end>':
      # remove the <start> and <end> tokens from the result string
      result = result.replace('<start> ', '')
      result = result.replace(' <end> ','')
      # remove the <start> and <end> tokens from the sentence string
      sentence = sentence.replace('<start> ', '')
      sentence = sentence.replace(' <end>', '')
      return  sentence, result

    # using the predicted word as the next decoder input
    dec_input = tf.expand_dims([predicted_id], 0)
  # remove the <start> and <end> tokens from the result string
  result = result.replace('<start> ', '')
  result = result.replace('<end>','')
  # remove the <start> and <end> tokens from the sentence string
  sentence = sentence.replace('<start> ', '')
  sentence = sentence.replace('<end>', '')
  

  
  # return the result string and the original sentence
  return sentence, result

print("text here")
print(chatbot("how are you today"))

print(chatbot('what is the weather outside'))

print(chatbot('what is the weather outside'))

print(chatbot(' how old '))

print(chatbot('can you play'))

print("finished")