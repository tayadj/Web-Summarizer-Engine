import tensorflow
import sklearn
import pandas
import string
import time
import re




class Preprocessor:

	def tokenize(self, language):

		tokenizer = tensorflow.keras.preprocessing.text.Tokenizer(filters = '')
		tokenizer.fit_on_texts(language)
		tensor = tokenizer.texts_to_sequences(language)
		tensor = tensorflow.keras.preprocessing.sequence.pad_sequences(tensor, padding = 'post')

		return tokenizer, tensor

	def process(self, text):

		text = re.sub(r'([{}])'.format(re.escape(string.punctuation)), r' \1 ', text)
		text = re.sub(r'\s+', ' ', text).strip()
		text = '<start> ' + text + ' <end>'
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
 
		BUFFER_SIZE = len(text_tensor_train)
		BATCH_SIZE = 5
		steps_per_epoch_train = len(text_tensor_train) // BATCH_SIZE
		steps_per_epoch_test = len(text_tensor_test) // BATCH_SIZE

		embedding_dimension = 256
		units = 1024 

		text_language_size = len(text_language) + 1
		summary_language_size = len(summary_language) + 1

		dataset_train = tensorflow.data.Dataset.from_tensor_slices((text_tensor_train, summary_tensor_train)).shuffle(BUFFER_SIZE)
		dataset_train = dataset_train.batch(BATCH_SIZE, drop_remainder = True)

		dataset_test = tensorflow.data.Dataset.from_tensor_slices((text_tensor_test, summary_tensor_test)).shuffle(BUFFER_SIZE)
		dataset_test = dataset_test.batch(BATCH_SIZE, drop_remainder = True)

		return dataset_train, dataset_test



