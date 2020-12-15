import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import logging
import logging.config
logging.config.fileConfig('logging.conf')
#LOG = logging.getLogger(__name__)
LOG = logging.getLogger()

# test logging code
LOG.debug('debug message')
LOG.info('info message')
LOG.warning('warn message')
LOG.error('error message')
LOG.critical('critical message')

RANDOM_SEED = 1234

PICKLE_ALLEN = "goodreads.pkl"
PICKLE_FLATTENED_TRAIN = "./goodreads_flattened_train.pkl"
PICKLE_FLATTENED_TEST = "./goodreads_flattened_test.pkl"


### load data
def load_data():
  df = pd.read_pickle(PICKLE_ALLEN)
  #train, test = train_test_split(df, test_size=0.2)

  LOG.debug("num_samples: {}".format(df.shape[0]))
  return df

'''

### (B) Data Preprocessing - tokenizer, etc
def find_star_rating_distribution():
  print("######### find_star_rating_distribution ###########")

  df = load_data()

  sns.countplot(df.overall)  ## this is automatically assigned to variable "plt"
  plt.xlabel('review rating')
  plt.show()


## FIND THE MAXLEN of all review summary
def find_maxlen_of_all_review_summary():
  print("######### FINDING MAXLEN OF ALL REVIEW SUMMARY ###########")
  mytokenizer = get_tokenizer()

  #df = load_data()
  df = load_data()

  token_lens = []
  for txt in df.summary:
    if isinstance(txt, str):
      tokens = mytokenizer.encode(txt, max_length=512)
      token_lens.append(len(tokens))

  sns.distplot(token_lens)
  plt.xlim([0, 60])
  plt.xlabel('Token count')
  plt.show()


## FIND THE MAXLEN of all reviewText
def find_maxlen_of_all_review_text():
  print("######### FINDING MAXLEN OF ALL REVIEW TEXT ###########")
  mytokenizer = get_tokenizer()

  #df = load_data()
  df = load_test_data_no_combine()

  token_lens = []
  for txt in df.reviewText:
    if isinstance(txt, str):
      tokens = mytokenizer.encode(txt, max_length=512)
      token_lens.append(len(tokens))

  sns.distplot(token_lens)
  plt.xlim([0, 1000])
  plt.xlabel('Token count')
  plt.show()
'''

###
def flatten_review_sentences(df_orig):
  sent = []
  spoil = []
  for i in tqdm(range(len(df_orig))):
    for j in range(len(df_orig.iloc[i]['review_sentences'])):
      sent.append(df_orig.iloc[i]['review_sentences'][j][1])   
      spoil.append(df_orig.iloc[i]['review_sentences'][j][0])
    # end for each sentence
  # end for each review


  df_new = pd.DataFrame(list(zip(sent, spoil)), columns =['sentence', 'has_spoiler']) 
  return df_new

###




######## MAIN #############
def main():

  ## load_data from json file
  df_orig = load_data()

  ## extract all the sentences
  df_new = flatten_review_sentences(df_orig)

  df_train, df_test = train_test_split(df_new, test_size=0.05, random_state=RANDOM_SEED)

  ## save to pkl
  df_train.to_pickle(PICKLE_FLATTENED_TRAIN)
  df_test.to_pickle(PICKLE_FLATTENED_TEST)

  



if __name__ == '__main__':
  #freeze_support()  ## this was suggested in relation to that "worker=" code in the dataset loader above; but probably not needed now.
  '''
  torch.manual_seed(1234)
  if torch.cuda.is_available():
      torch.set_default_tensor_type(torch.cuda.FloatTensor)
      torch.cuda.manual_seed_all(1234)
      LOG.debug(f'Running on GPU: {torch.cuda.get_device_name()}.')
  else:
      LOG.debug('Running on CPU.')
  '''
  main()
  #find_maxlen_of_all_review_text()
  #find_maxlen_of_all_review_summary()
  #find_star_rating_distribution()
