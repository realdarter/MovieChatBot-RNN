import tensorflow as tf

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
        output = self.fc(output)
        return output, hidden
  
    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            'vocab_inp_size': self.vocab_inp_size,  # Update to use vocab_inp_size attribute
            'embedding_dim': self.embedding_dim,
            'decoder_units': self.decoder_units,
            'batch_size': self.batch_size
        })
        return config



    @classmethod
    def from_config(cls, config):
        return cls(**config)
