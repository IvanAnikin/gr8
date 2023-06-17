import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the saved model
loaded_model = tf.keras.models.load_model('bot_model.h5')

# Preprocess the data
tokenizer = Tokenizer()

# Get the input shape of the model
input_shape = loaded_model.input_shape[1:]
maxlen = input_shape[0] if isinstance(input_shape, tuple) else input_shape
sample_shape = input_shape[1:] if isinstance(input_shape, tuple) else ()

while True:
    # Get user input
    user_input = input("User: ")

    if not user_input:  # Check if user input is empty
        print("Please provide a valid input.")
        continue

    # Preprocess the user input
    user_input_sequence = tokenizer.texts_to_sequences(user_input.split())

    if not user_input_sequence:  # Check if user input sequence is empty
        print("Please provide a valid input.")
        continue

    user_input_data = pad_sequences(user_input_sequence, padding='post', maxlen=maxlen, value=0)

    print("User input shape:", user_input_data.shape)
    print("Model input shapes:", loaded_model.input_shape)

    # Generate the bot's response
    response = loaded_model.predict([user_input_data, user_input_data])
    response_token_id = tf.argmax(response, axis=-1).numpy()[0][-1]

    # Convert token ID to word
    bot_response = tokenizer.index_word[response_token_id]

    print("GR8 Bot:", bot_response)