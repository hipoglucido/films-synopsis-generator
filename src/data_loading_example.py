import sqlite3
import pandas as pd
import settings
import os

df = pd.read_csv(filepath_or_buffer  = os.path.join(settings.RAW_DATA_DIR,'synopsis_genres.csv'),
                sep = '#',
                encoding = 'latin_1',
                index_col = 'ID',
                nrows = 100)
df.info()
print(df.head())