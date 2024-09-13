# Toxic-Comment-Analysis
![Screenshot 2024-09-13 194009](https://github.com/user-attachments/assets/7e217f17-ba56-47ae-99ab-7cfb1d10ca88)

![Screenshot 2024-09-13 195205](https://github.com/user-attachments/assets/abbc2d97-d6b5-4517-920a-bff01ef9eedc)


Importing Libraries and Loading Data

The code imports necessary libraries such as NumPy, Pandas, TensorFlow, and Keras.
It loads three CSV files: Data.csv, test.csv, and test_labels.csv.
The data is stored in Pandas DataFrames: full_data, test_data_X, and test_data_y.
Data Preprocessing

The code merges test_data_X and test_data_y into a single DataFrame test_dataframe based on the id column.
It removes rows with missing values (-1) from test_dataframe.
The code prints the number of observations in full_data and test_dataframe.
It deletes unnecessary DataFrames test_data_X and test_data_y.
Converting DataFrames to TensorFlow Datasets

The code converts full_data and test_dataframe into TensorFlow Datasets using tf.data.Dataset.from_tensor_slices.
It specifies the batch size as 16 and caches the data.
Text Vectorization

The code defines a TextVectorization layer with the following settings:
max_tokens: 100,000
standardize: lower and strip punctuation
output_mode: int
output_sequence_length: 1,800
It adapts the TextVectorization layer to the full_data comment text.
Model Definition

The code defines a Bi-LSTM model using the Keras Sequential API.
The model consists of the following layers:
TextVectorization layer
Embedding layer with 32 dimensions
Bidirectional LSTM layer with 32 units
Three Dense layers with ReLU activation and 256, 256, and 128 units, respectively
Final Dense layer with sigmoid activation and 6 units
The model is compiled with the Adam optimizer, binary cross-entropy loss, and binary accuracy metric.
Model Training

The code splits the training data into training and validation sets using take and skip methods.
It defines two callbacks: EarlyStopping and ReduceLROnPlateau.
The model is trained for 15 epochs with the specified callbacks and validation data.
Model Evaluation

The code evaluates the model on the test data using model.evaluate.
It plots the training and validation accuracy using Matplotlib.
Model Saving and Prediction

The code saves the model to a file named toxicity in TensorFlow format.
It defines a function to make predictions on new input text using the trained model.
Overall, this code implements a Bi-LSTM model for toxicity classification using TensorFlow and Keras. It loads and preprocesses the data, defines the model architecture, trains the model, evaluates its performance, and saves it for future use.
