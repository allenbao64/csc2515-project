from collections import defaultdict
import json
from tqdm import tqdm

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

from transformers import AutoTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

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



from roberta3_train import SpoilerClassifier
from roberta3_train import get_tokenizer
from roberta3_train import GoodreadsDataset
from roberta3_train import create_data_loader


# my data structures
RANDOM_SEED = 123
from roberta1_preproc import PICKLE_FLATTENED_TEST

from roberta3_train import COLUMN_NAMES
from roberta3_train import PRE_TRAINED_MODEL_NAME
from roberta3_train import MAX_LEN
from roberta3_train import BATCH_SIZE

BEST_MODEL_BIN = 'best_model_state.bin'  ## CHANGE THIS TO THE FILE YOU WANT



### load data

def load_data():
  df = pd.read_pickle(PICKLE_FLATTENED_TEST)

  LOG.debug("num_samples: {}".format(df.shape[0]))
  return df

'''

### (C) Dataset
class TestDataset(Dataset):

  def __init__(self, sentences, targets, tokenizer, max_len):
    self.sentences = sentences
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.sentences)
  
  def __getitem__(self, item):
    sentence = str(self.sentences[item])
    target = self.targets[item]

    encoding = self.tokenizer.encode_plus(
      sentence,
      add_special_tokens=True,
      return_token_type_ids=False,
      #pad_to_max_length=True,
      padding='max_length',
      truncation=True,
      max_length=self.max_len,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'sentence_text': sentence,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      #'targets': torch.tensor(target, dtype=torch.long)
      #'targets': torch.tensor(target, dtype=torch.float)  ## dropping "dtype=torch.long" to do linear regression
      'targets': torch.tensor(target, dtype=torch.float)  ## dropping "dtype=torch.long" to do linear regression
    }

###

def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = TestDataset(
    #sentences=df.summary.to_numpy(),
    sentences=df.combined.to_numpy(),                  ## df "combined" column is "summary"+"sentenceText"
    users=df.reviewerID.to_numpy(),
    products=df.itemID.to_numpy(),
    #targets=df.overall.to_numpy(),                   ## if linear regression, then just 1 target, not 5
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=0 ## num_workers=4
  )

'''

### INFERENCE
def infer(model, data_loader):
  model = model.eval()

  sigmoids = []
  y_hats = []
  y_labels = []

  with torch.no_grad():
    for d in tqdm(data_loader):
      input_ids = d["input_ids"]
      attention_mask = d["attention_mask"]
      targets = d["targets"]
      targets = targets.view(-1,1)  ## targets.shape (16) need to match outputs.shape (16,1) below; otherwise loss_fn complains below

      outputs = model(input_ids=input_ids, attention_mask=attention_mask)   ## outputs.shape = [batch, 1] = [8,1]

      preds = torch.zeros_like(outputs)
      ones = torch.ones_like(preds)
      preds = torch.where(outputs < 0, preds, ones)

      sigmoids.extend(torch.sigmoid(outputs))
      y_hats.extend(preds)
      y_labels.extend(targets)
    ## end for each batch
  ## end with torch.no_grad()
  sigmoids = torch.stack(sigmoids)
  y_hats = torch.stack(y_hats)
  y_labels = torch.stack(y_labels)
  return sigmoids.cpu().numpy(), y_hats.cpu().numpy(), y_labels.cpu().numpy()

  #return np.mean(losses)
## end infer()



######## MAIN #############
def main():
    ## load_data from json file
    df2 = load_data()

    #mytokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)\
    mytokenizer = get_tokenizer()

    # PyTorch data loader
    test_data_loader = create_data_loader(df2, mytokenizer, MAX_LEN, BATCH_SIZE)

    # load the trained model
    model = SpoilerClassifier()
    model.load_state_dict(torch.load(BEST_MODEL_BIN))

    # run inference
    sigmoids, y_hats, y_labels = infer(model, test_data_loader)
    print(classification_report(y_labels, y_hats))
    print(roc_auc_score(y_labels, y_hats))

## end main()



if __name__ == '__main__':
  #freeze_support()  ## this was suggested in relation to that "worker=" code in the dataset loader above; but probably not needed now.
  torch.manual_seed(1234)
  if torch.cuda.is_available():
      torch.set_default_tensor_type(torch.cuda.FloatTensor)
      torch.cuda.manual_seed_all(1234)
      LOG.debug(f'Running on GPU: {torch.cuda.get_device_name()}.')
  else:
      LOG.debug('Running on CPU.')

  main()





#########################################################
#########################################################
#########################################################

'''
def show_confusion_matrix(confusion_matrix):
  hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
  plt.ylabel('True sentiment')
  plt.xlabel('Predicted sentiment');
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
show_confusion_matrix(df_cm)
'''
