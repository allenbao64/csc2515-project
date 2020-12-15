import pandas as pd
from sklearn.model_selection import train_test_split

import logging
import logging.config
logging.config.fileConfig('logging.conf')
LOG = logging.getLogger()
# see which log level you're at
LOG.debug('debug message')
LOG.info('info message')
LOG.warning('warn message')
LOG.error('error message')
LOG.critical('critical message')

from r0_constants import RANDOM_SEED

from r0_constants import FILE3_FLATTENED
from r0_constants import FILE4_FLATTENED_TRAIN
from r0_constants import FILE5_FLATTENED_CV
from r0_constants import FILE6_FLATTENED_TEST


### load data
def load_flattened():
  flattened_df = pd.read_pickle(FILE3_FLATTENED)
  #train, test = train_test_split(df, test_size=0.2)

  LOG.debug("num_samples: {}".format(len(flattened_df)))
  return flattened_df



######## MAIN #############
def main():

  # load flattened sentences, with book title
  flattened_df = load_flattened()  #debug: 1M lines?

  # 90% train, 5% cv, 5% test
  df_train, df_cv_and_test = train_test_split(flattened_df, test_size=0.1, random_state=RANDOM_SEED)
  df_cv, df_test = train_test_split(df_cv_and_test, test_size=0.5, random_state=RANDOM_SEED)

  ## save to pkl
  df_train.to_pickle(FILE4_FLATTENED_TRAIN)
  df_cv.to_pickle(FILE5_FLATTENED_CV)
  df_test.to_pickle(FILE6_FLATTENED_TEST)

# end main()
  



if __name__ == '__main__':
  main()
