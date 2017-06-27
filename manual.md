## EXECUTION
The program was written in python 3 and these are the main libraries used: numpy, pandas, keras, tensorflow, nltk, scikit-learn.
All its fucntionalities must been executed from main.py like this:
python3 main.py
To decide which task to perform the user should make that explicitly modifying the end of the main.py.

## STAGES
The three main stages of this implementation are data loading, preprocessing, training and prediction. 
They are represented by functions that can be commented/uncommented at the bottom of the main.py file, depending on which task the user decides to use. The current state of the code is for the user to generate synopsis (PREDICTION stage).

## DATA LOADING
The first data this project builds on is the synopsis_genres.csv file, which has the url id, synopsis and genres for each film.

## PREPROCESSING
In this step all the files that are required for training the network are produced.
These vary depending on the desired vocabulary size, number of genres and if word2vec embeddings are going to be used.
Each of the network configurations that are explained in the report use different versions of the files. These files are included and are placed in the right folder for use the program. However they can be also generated again executing the function "generate_files()" in main.py.
The program will read the synopsis_genres.csv and produce the output according to settings.py, where the desired configuration must be written explicitly (under section ## PREPROCESSING).
If the user wants to generate a new embedding matrix, please donwload the model (.txt) version of it and put it into the /data/others folder. It is not included here since it is more than 1G size.

## TRAINING
For training a network the generated files needs are need. Training is possible writting the desired configuration in settings.py (under the section ## TRAINING )and uncommenting the train_networks() call in main.py. Again, the weights of the main networks trained in this project are provided. The weights names provide useful information. For example "LSTM_w2v0_v7000_g8_w-005-tloss5.3248-vloss5.2801" means, from left to right, that the LSTM network was trained without using word2vec embeddings, a vocabulary of 7000 words, 8 genres, during 5 epochs, with a training loss of 5.4 and validation loss of 5.2.

## PREDICTION
The provided code is ready to be executed in interface mode. One the user runs python3 main.py, a simple interface will be shown to the user and will guide him to the process of synopsis generation. For each generation, the user will be asked whether to use the greedy or beam search algorithm. If beam search is selected, the user will have to input the number of beams as well. Then the user will have to provide a set of genres. An optional step is to include some words for starting, with the aim of making the network continue what the user has written. If the user doesn't have a GPU, this each prediction may take up to 1 minute. If the user want to use another of the provided weights instead, the value of settings.WEIGHTS_PATH should be changed.

