import os
import data
import settings
#import model

import numpy as np

def test_generator():
    c = data.Generator()
    c.initialize()
    from time import sleep 
    t = 3000
    for i in range(1):
        a,b = c.generate().__next__()
        t-=1
        for i in range(a[0].shape[0]):
            print(c.to_genre(a[0][i]))
            print(c.to_synopsis(a[1][i]))
            print(c.to_synopsis(np.nonzero(b[i])[0]))
            print('_______________________________________')

            
def generate_files():
    preprocessor = data.Preprocessor()
    df = preprocessor.load_dataset()
    
    preprocessor.preprocess_synopses(df)
    preprocessor.preprocess_genres(df)
    preprocessor.build_indexes()
    preprocessor.generate_embedding_weights()
    preprocessor.filter_dataset()
    preprocessor.encode_genres()
    preprocessor.encode_synopses()
    preprocessor.save_data()

    
def train_network():
    network = model.Network()
    network.load_generator()
    network.build()
    network.load_weights()
    network.compile()
    network.train()


if __name__ == '__main__':
    #test_generator()
    #train_network()
    generate_files()
    #g= data.Generator()
    #g.load_preprocessed_data()
    #a = g.generate().__next__()
    #g.get_train_val_generators()