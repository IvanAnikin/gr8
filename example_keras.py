'''
import tensorflow as tf

import os
import yaml

dir_path = 'raw_data'
files_list = os.listdir(dir_path + os.sep)

questions = list()
answers = list()
for filepath in files_list:
    stream = open( dir_path + os.sep + filepath , 'rb')
    docs = yaml.safe_load(stream)
    conversations = docs['conversations']
    for con in conversations:
        if len( con ) > 2 :
            questions.append(con[0])
            replies = con[ 1 : ]
            ans = ''
            for rep in replies:
                ans += ' ' + rep
            answers.append( ans )
        elif len( con )> 1:
            questions.append(con[0])
            answers.append(con[1])

num_tokens = 10

encoder_inputs = tf.keras.layers.Input(shape=( None , ))
encoder_embedding = tf.keras.layers.Embedding( num_tokens, 200 , mask_zero=True) (encoder_inputs)
encoder_outputs , state_h , state_c = tf.keras.layers.LSTM( 200 , return_state=True )( encoder_embedding )
encoder_states = [ state_h , state_c ]

decoder_inputs = tf.keras.layers.Input(shape=( None ,  ))
decoder_embedding = tf.keras.layers.Embedding( num_tokens, 200 , mask_zero=True) (decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM( 200 , return_state=True , return_sequences=True )
decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )
decoder_dense = tf.keras.layers.Dense( num_tokens , activation=tf.keras.activations.softmax ) 
output = decoder_dense ( decoder_outputs )

model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output )
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy')

model.summary()



def make_inference_models():
    
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)
    
    decoder_state_input_h = tf.keras.layers.Input(shape=( 200 ,))
    decoder_state_input_c = tf.keras.layers.Input(shape=( 200 ,))
    
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding , initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    
    return encoder_model , decoder_model


'''



'''

import tensorflow as tf
import os
import yaml

# Load and preprocess the data
dir_path = 'raw_data'
files_list = os.listdir(dir_path + os.sep)

questions = list()
answers = list()
for filepath in files_list:
    stream = open(dir_path + os.sep + filepath, 'rb')
    docs = yaml.safe_load(stream)
    conversations = docs['conversations']
    for con in conversations:
        if len(con) > 2:
            questions.append(con[0])
            replies = con[1:]
            ans = ''
            for rep in replies:
                ans += ' ' + rep
            answers.append(ans)
        elif len(con) > 1:
            questions.append(con[0])
            answers.append(con[1])

# Preprocess the data further if needed (tokenization, padding, etc.)

# Define the model architecture
num_tokens = 10


encoder_inputs = tf.keras.layers.Input(shape=( None , ))
encoder_embedding = tf.keras.layers.Embedding( num_tokens, 200 , mask_zero=True) (encoder_inputs)
encoder_outputs , state_h , state_c = tf.keras.layers.LSTM( 200 , return_state=True )( encoder_embedding )
encoder_states = [ state_h , state_c ]


# ... Rest of the model architecture ...
decoder_inputs = tf.keras.layers.Input(shape=(None,))

decoder_dense = tf.keras.layers.Dense( num_tokens , activation=tf.keras.activations.softmax ) 
decoder_lstm = tf.keras.layers.LSTM( 200 , return_state=True , return_sequences=True )

decoder_inputs = tf.keras.layers.Input(shape=( None ,  ))
decoder_embedding = tf.keras.layers.Embedding( num_tokens, 200 , mask_zero=True) (decoder_inputs)
decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )

output = decoder_dense(decoder_outputs)



# Define the model
model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output)
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy')



# ------------------------------
#  DATA LOAD AND PREP

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Preprocess the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions + answers)
num_tokens = len(tokenizer.word_index) + 1  # Add 1 for the padding token

# Convert questions to sequences
encoder_sequences = tokenizer.texts_to_sequences(questions)
encoder_inputs_data = pad_sequences(encoder_sequences, padding='post')

# Convert answers to sequences with <START> and <END> tokens
decoder_input_sequences = tokenizer.texts_to_sequences(['<START> ' + answer for answer in answers])
decoder_inputs_data = pad_sequences(decoder_input_sequences, padding='post')

# Convert answers to sequences without <START> token and one-hot encode
decoder_target_sequences = tokenizer.texts_to_sequences([answer + ' <END>' for answer in answers])
decoder_target_data = tf.keras.utils.to_categorical(pad_sequences(decoder_target_sequences, padding='post'), num_tokens)


# ------------------------------




# Train the model
model.fit(x=[encoder_inputs_data, decoder_inputs_data], y=decoder_target_data, batch_size=32, epochs=10)


'''
