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

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            'vocab_size': self.vocab_size,  # Use self.vocab_size instead of vocab_size
            'embedding_dim': self.embedding_dim,
            'encoder_units': self.encoder_units,
            'batch_size': self.batch_size })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
