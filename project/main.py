import tensorflow
import sklearn
import pandas
import string
import numpy
import time
import re



class Model:

	def __init__(self, config):

		self.batch_size = config.get('batch_size', 64)
		self.embedding_dimension = config.get('embedding_dimension', 256)
		self.units = config.get('units', 1024)
		self.start_token = config.get('start_token', '<start>')
		self.end_token = config.get('end_token', '<end>')
		self.data_path = config.get('data_path', './data/data.csv')
		self.model_path = config.get('model_path', './data')

		self.processor = self.Processor(self.batch_size, self.embedding_dimension, self.units, self.start_token, self.end_token, self.data_path, self.model_path)



	class Processor:

		def __init__(self, batch_size, embedding_dimension, units, start_token, end_token, data_path, model_path):

			self.batch_size = batch_size
			self.embedding_dimension = embedding_dimension
			self.units = units
	
			self.start_token = start_token
			self.end_token = end_token

			self.data_path = data_path
			self.model_path = model_path

		def tokenize(self, language):

			tokenizer = tensorflow.keras.preprocessing.text.Tokenizer(filters = '')
			tokenizer.fit_on_texts(language)
			tensor = tokenizer.texts_to_sequences(language)
			tensor = tensorflow.keras.preprocessing.sequence.pad_sequences(tensor, padding = 'post')

			return tokenizer, tensor

		def process(self, text):

			text = re.sub(r'([{}])'.format(re.escape(string.punctuation)), r' \1 ', text)
			text = re.sub(r'\s+', ' ', text).strip()
			text = self.start_token + ' ' + text + ' ' + self.end_token
			text = text.lower()

			return text

		def load_data(self, path = './data.csv'):

			data_raw = pandas.read_csv(path)

			data = data_raw
			data['Text'] = data['Text'].apply(self.process)
			data['Summary'] = data['Summary'].apply(self.process)

			return data

		def load_dataset(self, path = './data.csv'):

			data = self.load_data(path)
			text_language = data['Text']
			summary_language = data['Summary']

			text_tokenizer, text_tensor = self.tokenize(text_language)
			summary_tokenizer, summary_tensor = self.tokenize(summary_language)
			text_tensor_train, text_tensor_test, summary_tensor_train, summary_tensor_test = sklearn.model_selection.train_test_split(text_tensor, summary_tensor, test_size = 0.2)
 
			buffer_size = len(text_tensor_train)
			batch_size = 1
			steps_per_epoch_train = len(text_tensor_train) // batch_size
			steps_per_epoch_test = len(text_tensor_test) // batch_size

			embedding_dimension = 256
			units = 1024 

			text_language_size = len(text_tokenizer.word_index) + 1
			summary_language_size = len(summary_tokenizer.word_index) + 1

			dataset_train = tensorflow.data.Dataset.from_tensor_slices((text_tensor_train, summary_tensor_train)).shuffle(buffer_size)
			dataset_train = dataset_train.batch(batch_size, drop_remainder = True)

			dataset_test = tensorflow.data.Dataset.from_tensor_slices((text_tensor_test, summary_tensor_test)).shuffle(buffer_size)
			dataset_test = dataset_test.batch(batch_size, drop_remainder = True)

			return text_tokenizer, summary_tokenizer, dataset_train, dataset_test, buffer_size, batch_size, steps_per_epoch_train, steps_per_epoch_test, embedding_dimension, units, text_language_size, summary_language_size



	class Encoder(tensorflow.keras.Model):

		def __init__(self, vocabulary_size, embedding_dimension, encoder_units, batch_size):

			super(Model.Encoder, self).__init__()

			self.vocabulary_size = vocabulary_size
			self.embedding_dimension = embedding_dimension
			self.encoder_units = encoder_units
			self.batch_size = batch_size

			self.embedding = tensorflow.keras.layers.Embedding(vocabulary_size, embedding_dimension)
			self.GRU = tensorflow.keras.layers.GRU \
			(
				encoder_units,
				return_sequences = True,
				return_state = True,
				recurrent_initializer = 'glorot_uniform'
			)

		def call(self, input, hidden):

			input = self.embedding(input)
			output, state = self.GRU(input, initial_state = hidden)

			return output, state

		def initialize_hidden_state(self):

			return tensorflow.zeros((self.batch_size, self.encoder_units))



	class Decoder(tensorflow.keras.Model):

		def __init__(self, vocabulary_size, embedding_dimension, decoder_units, batch_size, attention = None):

			super(Model.Decoder, self).__init__()

			self.vocabulary_size = vocabulary_size
			self.embedding_dimension = embedding_dimension
			self.decoder_units = decoder_units
			self.batch_size = batch_size
			self.attention = attention

			self.embedding = tensorflow.keras.layers.Embedding(vocabulary_size, embedding_dimension)
			self.dense = tensorflow.keras.layers.Dense(vocabulary_size)
			self.GRU = tensorflow.keras.layers.GRU \
			(
				decoder_units,
				return_sequences = True,
				return_state = True,
				recurrent_initializer = 'glorot_uniform'
			)

		def call(self, input, hidden, encoder_output):
		
			input = self.embedding(input)

			attention_weights = None

			if self.attention:

				context_vector, attention_weights = self.attention(hidden, encoder_output)
				input = tensorflow.concat([tensorflow.expand_dims(context_vector, 1), input], axis = -1)

			output, state = self.GRU(input, initial_state = hidden)
			output = tensorflow.reshape(output, (-1, output.shape[2]))
			predictions = self.dense(output)			

			return predictions, state, attention_weights



	def loss_function(self, observation, expectation):

		loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = 'none')
		loss_ = loss(observation, expectation)

		mask = tensorflow.math.logical_not(tensorflow.math.equal(observation, 0))
		mask = tensorflow.cast(mask, dtype = loss_.dtype)
		loss_ *= mask

		return tensorflow.reduce_mean(loss_)



	def train_engine(self):

		optimizer = tensorflow.keras.optimizers.Adam()	

		@tensorflow.function
		def train_step(text, summary, encoder_hidden, encoder, decoder):

			text_tokenizer, summary_tokenizer, dataset_train, dataset_test, buffer_size, batch_size, steps_per_epoch_train, steps_per_epoch_test, embedding_dimension, units, text_language_size, summary_language_size = self.processor.load_dataset()
		
			loss = 0

			with tensorflow.GradientTape() as tape:

				encoder_output, encoder_hidden = encoder(text, encoder_hidden)
				decoder_hidden = encoder_hidden
				decoder_input = tensorflow.expand_dims([summary_tokenizer.word_index[self.start_token]] * batch_size, 1)

				for time in range(1, summary.shape[1]):

					predictions, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)
					loss += self.loss_function(summary[:, time], predictions)
					decoder_input = tensorflow.expand_dims(summary[:, time], 1)

				batch_loss = (loss / int(summary.shape[1]))
				variables = encoder.trainable_variables + decoder.trainable_variables
				gradients = tape.gradient(loss, variables)
				optimizer.apply_gradients(zip(gradients, variables))

			return batch_loss

		return train_step



	def test_engine(self, text, summary, encoder_hidden, encoder, decoder):

		text_tokenizer, summary_tokenizer, dataset_train, dataset_test, buffer_size, batch_size, steps_per_epoch_train, steps_per_epoch_test, embedding_dimension, units, text_language_size, summary_language_size = self.processor.load_dataset()

		loss = 0

		encoder_output, encoder_hidden = encoder(text, encoder_hidden)
		decoder_hidden = encoder_hidden
		decoder_input = tensorflow.expand_dims([summary_tokenizer.word_index[self.start_token]] * batch_size, 1)

		for time in range(1, summary.shape[1]):

			predictions, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)
			loss += self.loss_function(summary[:, time], predictions)
			decoder_input = tensorflow.expand_dims(summary[:, time], 1)

		batch_loss = (loss / int(summary.shape[1]))

		return batch_loss



	def learn(self, epochs = 10):

		text_tokenizer, summary_tokenizer, dataset_train, dataset_test, buffer_size, batch_size, steps_per_epoch_train, steps_per_epoch_test, embedding_dimension, units, text_language_size, summary_language_size = self.processor.load_dataset()

		encoder = self.Encoder(text_language_size, embedding_dimension, units, batch_size)
		decoder = self.Decoder(summary_language_size, embedding_dimension, units, batch_size)

		train_function = self.train_engine()
		test_function = self.test_engine
		train_loss = []
		test_loss = []

		for epoch in range(epochs):

			start = time.time()

			encoder_hidden = encoder.initialize_hidden_state()
			total_loss_train = 0

			for (batch, (text, summary)) in enumerate(dataset_train.take(steps_per_epoch_train)):

				batch_loss = train_function(text, summary, encoder_hidden, encoder, decoder)
				total_loss_train += batch_loss

				print('Train Epoch {} | Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss))

			encoder_hidden = encoder.initialize_hidden_state()
			total_loss_test = 0

			for (batch, (text, summary)) in enumerate(dataset_test.take(steps_per_epoch_test)):

				batch_loss = test_function(text, summary, encoder_hidden, encoder, decoder)
				total_loss_test += batch_loss

				print('Test Epoch {} | Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss))

			train_loss.append(total_loss_train / steps_per_epoch_train)
			test_loss.append(total_loss_test / steps_per_epoch_test)

			print('Epoch {} | Train Loss {:.4f}, Test Loss {:.4f}'.format(epoch + 1, train_loss[-1], test_loss[-1]))
			print('Time taken for epoch {} sec\n'.format(time.time() - start))

		return encoder, decoder, train_loss, test_loss



	def summarize(self, sentence, encoder, decoder):

		data = self.processor.load_data('./data.csv')
		text_language = data['Text']
		summary_language = data['Summary']
		text_tokenizer, text_tensor = self.processor.tokenize(text_language)
		summary_tokenizer, summary_tensor = self.processor.tokenize(summary_language)
		text_tokenizer, summary_tokenizer, dataset_train, dataset_test, buffer_size, batch_size, steps_per_epoch_train, steps_per_epoch_test, embedding_dimension, units, text_language_size, summary_language_size = self.processor.load_dataset()
		max_length_summary = summary_tensor.shape[1] 
		max_length_text =  text_tensor.shape[1]

		attention_plot = numpy.zeros((max_length_summary, max_length_text))

		sentence = self.processor.process(sentence)

		inputs = []
		for word in sentence.split(' '):

			try:

				inputs.append(text_tokenizer.word_index[word])

			except KeyError:

				continue

		inputs = tensorflow.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_text, padding='post')
		inputs = tensorflow.convert_to_tensor(inputs)

		result = ''

		hidden = [tensorflow.zeros((1, units))]
		encoder_output, encoder_hidden = encoder(inputs, hidden)

		decoder_hidden = encoder_hidden
		decoder_input = tensorflow.expand_dims([summary_tokenizer.word_index[self.start_token]], 0)

		for time in range(max_length_summary):
	
			predictions, decoder_hidden, attention_weights = decoder(decoder_input, decoder_hidden, encoder_output)

			prediction = tensorflow.argmax(predictions[0]).numpy()
			result += summary_tokenizer.index_word[prediction] + ' '

			if summary_tokenizer.index_word[prediction] == self.end_token:
			
				return result, sentence

			decoder_input = tensorflow.expand_dims([prediction], 0)

		return result, sentence



def demo(epochs = 10):

	config = {
		'batch_size': 1,
		'embedding_dimension': 128,
		'units': 512,
		'start_token': '<start>',
		'end_token': '<end>'
	}

	model = Model(config)
	encoder, decoder, train_loss, test_loss = model.learn(epochs)

	data = pandas.read_csv('./data.csv')['Text']
	result = [model.summarize(text, encoder, decoder) for text in data]

	for result_ in result:

		print(result_)