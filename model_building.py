import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, encoder_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.encoder_units = encoder_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.gru = tf.keras.layers.GRU(self.encoder_units, 
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.encoder_units))
    
    @classmethod
    def from_config(cls, config):
        return cls(config['vocab_size'], config['embedding_dim'], config['encoder_units'], config['batch_size'])

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, decoder_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.decoder_units = decoder_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.gru = tf.keras.layers.GRU(self.decoder_units, 
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, hidden = self.gru(x, initial_state=hidden)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = tf.nn.softmax(self.fc(output))
        return x, hidden
    
    @classmethod
    def from_config(cls, config):
        return cls(config['vocab_size'], config['embedding_dim'], config['decoder_units'], config['batch_size'])

def build_encoder(vocab_size, embedding_dim, units, batch_size):
    return Encoder(vocab_size, embedding_dim, units, batch_size)

def build_decoder(vocab_size, embedding_dim, units, batch_size):
    return Decoder(vocab_size, embedding_dim, units, batch_size)
