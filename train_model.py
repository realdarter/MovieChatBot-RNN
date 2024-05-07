import tensorflow as tf

def create_dataset(x, y, batch_size=32):
    data = tf.data.Dataset.from_tensor_slices((x, y))
    data = data.shuffle(1028)
    data = data.batch(batch_size, drop_remainder=True)
    data = data.prefetch(tf.data.experimental.AUTOTUNE)
    return data

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

@tf.function
def train_step(inputs, target, enc_hidden, encoder, decoder, optimizer, target_tokenizer):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inputs, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([target_tokenizer.word_index['<start>']] * inputs.shape[0], 1)
        for t in range(1, target.shape[1]):
            predictions, dec_hidden = decoder(dec_input, dec_hidden)
            loss += loss_function(target[:, t], predictions)
            dec_input = tf.expand_dims(target[:, t], 1)
    batch_loss = (loss / int(target.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss

@tf.function 
def test_step(inputs, target, enc_hidden, encoder, decoder, target_tokenizer):
    loss = 0
    enc_output, enc_hidden = encoder(inputs, enc_hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_tokenizer.word_index['<start>']] * inputs.shape[0], 1)
    for t in range(1, target.shape[1]):
        predictions, dec_hidden = decoder(dec_input, dec_hidden)
        loss += loss_function(target[:, t], predictions)
        dec_input = tf.expand_dims(target[:, t], 1)
    batch_loss = (loss / int(target.shape[1]))
    return batch_loss

def train_model(encoder, decoder, train_dataset, test_dataset, optimizer, EPOCHS, target_tokenizer):
    old_test_loss = 1000000
    for epoch in range(EPOCHS):
        for (batch, (inputs, target)) in enumerate(train_dataset):
            enc_hidden = encoder.initialize_hidden_state()
            batch_loss = train_step(inputs, target, enc_hidden, encoder, decoder, optimizer, target_tokenizer)
        for (batch, (inputs, target)) in enumerate(test_dataset):
            enc_hidden = encoder.initialize_hidden_state()
            batch_loss = test_step(inputs, target, enc_hidden, encoder, decoder, target_tokenizer)
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
