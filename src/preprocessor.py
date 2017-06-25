
import settings

import os
import numpy as np
import pandas as pd
import re

from sklearn.preprocessing import MultiLabelBinarizer
from collections import defaultdict
from sklearn.externals import joblib
from nltk.tag import pos_tag
from time import strftime

class Preprocessor():

    def __init__(self):
        """
        Loads the films csv (genres and synopsis)
        Preprocesses both of them and keeps them in memory
        Initialises
        """
        #Attributes list (initialised afterwards in this method)
        self.index_to_word = None #{ 3:'hola', 3002: 'cielo' ...
        self.word_to_index = None #{'hola': 3, 'cielo': 3002, ...
        self.mlb = None
        self.encoded_genres = None
        self.synopses = None
        self.count = 0

    
    def build_indexes(self):
        settings.logger.info("Building word indexes")
        #print(str(unique).encode('latin1'))
        self.word_to_index = {}
        self.index_to_word = {}

        for i, word in enumerate(self.vocabulary):
            self.word_to_index[word]=i
            self.index_to_word[i]=word
        assert len(self.word_to_index) == len(self.index_to_word)
        joblib.dump(self.word_to_index, os.path.join(settings.OTHERS_DIR, strftime("%Y%m%dT%H%M%S")+'_word_to_index.pkl'))
        joblib.dump(self.index_to_word, os.path.join(settings.OTHERS_DIR, strftime("%Y%m%dT%H%M%S")+'_index_to_word.pkl'))
        settings.logger.info("Saved index dictionaries for "+str(len(self.word_to_index))+" words in "+settings.OTHERS_DIR)
        
    def preprocess_synopses(self, df):
        settings.logger.info("Preprocessing synopses...")

        #Keep the synopsis as a list
        self.synopses = df['Synopsis'].map(self.clean_text)

        settings.logger.info("Tokenizing synopses...")
        self.synopses = self.synopses.map(self.tokenize)

        self.synopses = list(self.synopses.values)
         
        from collections import defaultdict

        settings.logger.info("Counting word frequency...")
        word_freqs = defaultdict(int)
        for synopsis in self.synopses:
            for word in synopsis:
                word_freqs[word] += 1
                
        word_freqs = list(word_freqs.items())
        most_frequent = sorted(word_freqs, key = lambda x: x[1], reverse = True)
        settings.logger.info("Most frequent words: " + str(most_frequent[:10]))
        self.vocabulary = [w[0] for w in most_frequent][:settings.VOCABULARY_SIZE]

        self.vocabulary[-1] = settings.UNKNOWN_TOKEN
        self.vocabulary[-2] = settings.PAD_TOKEN
        if settings.EOS_TOKEN not in self.vocabulary:
            self.vocabulary[-3] = settings.EOS_TOKEN
        settings.logger.info("Only "+str(len(self.vocabulary))+" words will be considered (VOCABULARY_SIZE)")

        #Substitute any unkown word with <unk> token inside the function.
        self.count = 0
        self.total = len(self.synopses)
        def map_unkown_tokens(obj, synopsis):
            new_synopsis = []
            for word in synopsis:
                if word in self.vocabulary:
                    new_synopsis.append(word)
                else:
                    new_synopsis.append(settings.UNKNOWN_TOKEN)
            obj.count += 1
            if obj.count % 100 == 0:
                settings.logger.info(str(100*obj.count/obj.total)[:4]+'% completed...')
            return new_synopsis
        settings.logger.info("Mapping unkown tokens...")
        #self.synopses = [map_unkown_tokens(synopsis) for synopsis in self.synopses]
        self.synopses = list(pd.Series(self.synopses).map(lambda x: map_unkown_tokens(self,x)))

    ''' Receives a string.
       Returns that same string after being preprocessed.'''

    def clean_text(self, text):
        # Handle (...)
        #text = 'asdas 9999 asdasdsd , hola... ! 787 8 '
        text_in_paren = re.findall("\([^\)]*\)", text)
        if text_in_paren:
            for del_text in text_in_paren:
                text = text.replace(del_text, '')
        # Handle digits
        digits = re.findall(r'\d+', text)
        if digits:
            for digit in digits:
                text = text.replace(digit, settings.DIGIT_TOKEN)
        # Remove puntuaction
        # text = "".join(c for c in text if c not in ('¡','!','¿','?', ':', ';'))
        text = re.sub(r'[^a-zA-Z\.áéíóúÁÉÍÓÚüÜñÑ]', ' ', text)
        # Remove extra spaces that were left when cleaning
        text = re.sub(r'\s+', ' ', text)
        '''
        #text = text.lower()
        text_tags = pos_tag(text.split())
        final = ""
        for word, pos in text_tags:
            if pos == "NPP" or word == settings.DIGIT_TOKEN:
                final += " " + word
            else:
                final += " " + word.lower()
        self.count +=1
        if self.count % 1000 == 0:
            settings.logger.info(self.count)
        '''
        return text
        
    def filter_dataset(self):
        """
        This is meant to be the last preprocessing step. It reduces the
        dataset and should be called after preprocess_synopsis and
        preprocess_genres
        """
        filtered_genres, filtered_synopses = [], []
        for genres, synopsis in zip(self.genres, self.synopses):
            known_words = len(synopsis) - synopsis.count(settings.UNKNOWN_TOKEN)
            if known_words / len(synopsis) < settings.MINIMUM_KNOWN_PERC_TOKENS_PER_SYNOPSIS:
                continue
            if len(genres) == 0:
                continue
            filtered_genres.append(genres)
            filtered_synopses.append(synopsis)
        self.genres = filtered_genres
        self.synopses = filtered_synopses    
     
        #settings.logger.info("Total tokens in corpus after preprocessing : "+str(corpus_tokens_count))
        
    def save_data(self):
        assert len(self.genres) == len(self.synopses)
        films_preprocessed = [self.encoded_genres, self.encoded_synopses]
        filepath = os.path.join(settings.DATA_DIR,strftime("%Y%m%dT%H%M%S_")+str(self.encoded_genres.shape[0])+"_preprocessed_films.pkl")
        #print(films_preprocessed)
        joblib.dump(films_preprocessed, filepath)
        settings.logger.info(str(len(self.encoded_genres))+" preprocessed films data saved to "+filepath) 
        
    
    def encode_genres(self):   
        #Preprocess genres (multilabel)
        self.encoded_genres = self.mlb.transform(self.genres)
        settings.logger.info(str(len(self.mlb.classes_))+" different genres found:"+str(self.mlb.classes_)[:100]+"...")
        #I am not sure when to use .encode('latin1')
        settings.logger.debug('Genres vector shape: '+str(self.encoded_genres.shape)) 
        settings.logger.info('Genres encoded')
    
    def encode_synopses(self):
        self.encoded_synopses = []
        for synopsis in self.synopses:
            encoded_synopsis = []
            for word in synopsis:
                encoded_synopsis.append(self.word_to_index[word])
            self.encoded_synopses.append(encoded_synopsis)

        settings.logger.info('Synopses encoded')
    
    def load_dataset(self):
        import pandas as pd
        if settings.USE_SMALL_DATASET:
            nrows = 4000
        else:
            nrows = None
        df = pd.read_csv(filepath_or_buffer  = os.path.join(settings.DATA_DIR,'synopsis_genres.csv'),sep = '#',encoding = 'latin_1',index_col = 'ID', nrows = nrows)
        df = df[df['Synopsis'].notnull() & df['Genre'].notnull()]
        settings.logger.info(str(df.info()))
        #settings.logger.info(str(df[['Genre','Synopsis']][:5]).encode('latin1'))
        return df
        
    def generate_embedding_weights(self):        
        settings.logger.info('Loading Word2Vec model from '+settings.WORD2VEC_MODEL_PATH)
        if settings.USE_SMALL_WORD2VEC:
            nrows = 100000
        else:
            nrows = None
        model = pd.read_csv(settings.WORD2VEC_MODEL_PATH, sep = ' ', header = None, \
                            index_col = 0, skiprows = 1, nrows = nrows)
        embedding_rows = len(self.vocabulary) + 1 # adding 1 to account for 0th index (for masking)
        settings.logger.info('Generating embedding weights matrix for '+str(embedding_rows)+' words...')
        embedding_weights = np.zeros((embedding_rows,settings.EMBEDDING_DIM))
        count = 0
        for index, word in self.index_to_word.items():
            #print(self.index_to_word[index],word)
            try:
                embedding_weights[index,:] = model.loc[word].values
            except KeyError:
                try:
                    embedding_weights[index,:] = model.loc[word.title()].values
                    settings.logger.warning(self.index_to_word[index]+' ('+word+') will take the embedding of '+word.title())
                except KeyError:
                    if 'digito' in self.index_to_word[index]:
                        embedding_weights[index,:] = model.loc[settings.DIGIT_TOKEN].values
                        settings.logger.warning(self.index_to_word[index]+' ('+word+') will take the embedding of '+settings.DIGIT_TOKEN)
                    else:  
                        settings.logger.warning(self.index_to_word[index]+' ('+word+') not found in word2vec')
                        count += 1

        #embedding_weights[embedding_rows-1, :] = settings.EOS_TOKEN
        settings.logger.info(str(count)+" tokens represented with zeros in the weight matrix "+str(embedding_weights.shape))
        filepath = os.path.join(settings.OTHERS_DIR,strftime("%Y%m%dT%H%M%S_")+str(embedding_weights.shape[0])+"_embedding_weights.pkl")
        joblib.dump(embedding_weights, filepath)
        settings.logger.info("Saved weights matrix in "+filepath)
        
    def preprocess_genres(self, df):
        
        self.genres = list(df['Genre'].map(lambda x: x.split('|')))

            
        genre_freqs = defaultdict(int)
        for genres_ in self.genres:
            for genre in genres_:
                genre_freqs[genre] += 1
        
        genre_freqs = list(genre_freqs.items())        
        most_frequent = sorted(genre_freqs, key = lambda x: x[1], reverse = True)
        settings.logger.info("Most frequent genres: " + str(most_frequent[:10]))
        
        knwown_genres = [g[0] for g in most_frequent][:settings.MAX_GENERES]
        
        settings.logger.info("Only "+str(len(knwown_genres))+" genres will be considered (MAX_GENERES)")
        
        de