

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import yaml

# Load and preprocess the data
dir_path = 'input/raw_data'
files_list = os.listdir(dir_path + os.sep)

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define the conversation data
categories = ['science']
conversations = [
    ['what are the laws of thermodynamics', 'i\'m not a physicist, but I think this has something to do with heat, entropy, and conservation of energy, right?'],
    ['what disease does a carcinogen cause', 'cancer.'],
    ['what is a wavelength', 'wavelength is the inverse of frequency.'],
    ['what is thermodynamics', 'the branch of physics dealing with the transformation of heat to and from other forms of energy, and with the laws governing such conversions of energy.'],
    ['what is chemistry', 'the science of mixing chemicals.'],
    ['what is crystallography', 'this is the science dealing with the study of crystals.'],
    ['what is avogadro s number', 'it is the number of molecules per mole. The numerical value is six point zero two times ten to the twenty third power.'],
    ['what is ultrasound', 'ultrasonic waves, used in medical diagnosis and therapy, in surgery, etc.'],
    ['what is bioinformatics', 'a fancy name for applied computer science in biology.'],
    ['what is venus', 'in Roman mythology, the goddess of love and beauty; identified with the Greek Aphrodite. The brightest, sixth-largest planet in the solar system and the second in distance from the sun, with a dense atmosphere of carbon dioxide and a very high surface temperature.'],
    ['what is ichthyology', 'we talk about this when we study fishes.'],
    ['what is h2o', 'h is to o as o is to v.'],
    ['what is cytology', 'the study of cells.'],
    ['what is cytology', 'well, from what I can recall it is the study of cells.'],
    ['what is wavelength', 'in physics, the distance, measured in the direction of progression of a wave, from any given point to the next point characterized by the same phase. Or is could be looked at as a way of thinking.'],
    ['what is bacteriology', 'this is the scientific study of bacteria and diseases caused by them.'],
    ['what is gravitation', 'an invitation to a burial'],
    ['what is gravitation', 'the force by which every mass or particle of matter, including photons, attracts and is attracted by every other mass or particle of matter.'],
    ['we are on the same wavelength', 'and the same frequency.'],
    ['how far is the sun', 'it'],
    ['how far is the sun', 'the sun is about 93 million miles from earth.'],
    ['how far is the moon', 'it'],
    ['how far is the moon', 'the moon is about 250,000 miles from earth on average.'],
    ['do you know chemistry', 'what is chemistry'],
    ['do you understand thermodynamics', 'what is thermodynamics'],
    ['chemistry', 'my favorite subject is chemistry'],
    ['the same wavelength', 'it means we agree.'],
    ['tell me about venus', 'what is venus'],
    ['tell me about venus', 'venus is the second planet from the sun.']
]

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
