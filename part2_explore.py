import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import json
import random

'''
with open('goodreads_reviews_spoiler.json') as f:
    lines = f.read().splitlines()
df = pd.DataFrame(lines)
df.columns = ['json_element']
df = df.sample(frac=0.2, random_state=1)
df['json_element'].apply(json.loads) #9.70
df = pd.json_normalize(df['json_element'].apply(json.loads)) #42.36
df.to_pickle('goodreads.pkl')
'''

df = pd.read_pickle('goodreads.pkl')
train, test = train_test_split(df, test_size=0.2)
