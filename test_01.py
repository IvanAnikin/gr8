


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import yaml

# Load and preprocess the data
dir_path = 'input/raw_data'
files_list = os.listdir(dir_path + os.sep)




input_dir = 'input/Croatiannonprofessionalwritten'



# LOAD INPUT DATA  ---------


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

questions = []
answers = []
for filepath in files_list:
    stream = open(os.path.join(dir_path, filepath), 'rb')
    docs = yaml.safe_load(stream)
    conversations = docs['conversations']
    for con in conversations:
        if len(con) > 1:
            questions.append(con[0])
            answers.append(con[1])

# Extract questions and answers from conversations
questions = [conv[0] for conv in conversations]
answers = [conv[1] for conv in conversations]


# Preprocess the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)
num_tokens = len(tokenizer.word_index) + 1  # Add 1 for the padding token

# Convert questions to sequences
encoder_sequences = tokenizer.texts_to_sequences(questions)
encoder_inputs_data = pad_sequences(encoder_sequences, padding='post')

# Convert answers to sequences with <START> and <END> tokens
decoder_input_sequences = tokenizer.texts_to_sequences(['<START> ' + answer + ' <END>' for answer in answers])
decoder_inputs_data = pad_sequences(decoder_input_sequences, padding='post')

# Convert answers to sequences without <START> token and one-hot encode
decoder_target_sequences = tokenizer.texts_to_sequences([answer + ' <END>' for answer in answers])
decoder_target_data = tf.keras.utils.to_categorical(pad_sequences(decoder_target_sequences, padding='post'), num_tokens)


# Define the model architecture
encoder_inputs = tf.keras.layers.Input(shape=(None,))
encoder_embedding = tf.keras.layers.Embedding(num_tokens, 200, mask_zero=True)(encoder_inputs)
encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(200, return_state=True)(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = tf.keras.layers.Input(shape=(None,))
decoder_embedding = tf.keras.layers.Embedding(num_tokens, 200, mask_zero=True)(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(200, return_state=True, return_sequences=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = tf.keras.layers.Dense(num_tokens, activation=tf.keras.activations.softmax)
output = decoder_dense(decoder_outputs)

# Define the model
model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy')

# Train the model
model.fit(x=[encoder_inputs_data, decoder_inputs_data], y=decoder_target_data, batch_size=32, epochs=100)



# Save the trained model
model.save('bot_model.h5')



while True:
    # Get user input
    user_input = input("User: ")

    # Preprocess the user input
    user_input_sequence = tokenizer.texts_to_sequences([user_input])
    user_input_data = pad_sequences(user_input_sequence, padding='post')

    # Generate the bot's response
    response = model.predict([user_input_data, user_input_data])
    response_token_id = np.argmax(response, axis=-1)[0][-1]

    # Convert token ID to word
    bot_response = tokenizer.index_word[response_token_id]

    print("GR8 Bot:", bot_response)