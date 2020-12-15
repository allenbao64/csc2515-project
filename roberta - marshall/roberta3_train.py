
from collections import defaultdict
import json
from tqdm import tqdm

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

#from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup  #, BertConfig
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

# my data structures
RANDOM_SEED = 123

from roberta1_preproc import PICKLE_FLATTENED_TRAIN


# save model to file
BEST_MODEL_BIN = 'best_model_state.bin'

COLUMN_NAMES = ["user_id", "timestamp", "review_sentences", "rating", "has_spoiler", "book_id", "review_id"]

#PRE_TRAINED_MODEL_NAME = 'bert-base-cased'  ## tune; can try distill-bert
#PRE_TRAINED_MODEL_NAME = 'distilroberta-base'
PRE_TRAINED_MODEL_NAME = 'roberta-base'  

MAX_LEN = 40  ## tune
BATCH_SIZE = 16  ## tune




### load data
def load_data():
  df = pd.read_pickle(PICKLE_FLATTENED_TRAIN)

  LOG.debug("num_samples: {}".format(df.shape[0]))
  return df


def load_data_DEBUG():
  df = pd.read_pickle(PICKLE_FLATTENED_TRAIN)

  df2 = df.head(100)

  LOG.debug("########## YOU ONLY HAVE {} SAMPLES !!! ##########".format(df2.shape[0]))
  LOG.debug("########## YOU ONLY HAVE {} SAMPLES !!! ##########".format(df2.shape[0]))
  LOG.debug("num_samples: {}".format(df2.shape[0]))
  LOG.debug("########## YOU ONLY HAVE {} SAMPLES !!! ##########".format(df2.shape[0]))
  LOG.debug("########## YOU ONLY HAVE {} SAMPLES !!! ##########".format(df2.shape[0]))
  return df2


### tokenzier
  # https://huggingface.co/transformers/model_doc/auto.html#autotokenizer 
  #>>> from transformers import AutoTokenizer
  #>>> # Download vocabulary from huggingface.co and cache.
  #>>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
  #>>> # Download vocabulary from huggingface.co (user-uploaded) and cache.
  #>>> tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-german-cased')
  #>>> # If vocabulary files are in a directory (e.g. tokenizer was saved using `save_pretrained('./test/saved_model/')`)
  #>>> tokenizer = AutoTokenizer.from_pretrained('./test/bert_saved_model/')

  # https://huggingface.co/transformers/model_doc/roberta.html#robertatokenizerfast
  #>>> from transformers import RobertaTokenizerFast
  #>>> tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
  #>>> tokenizer("Hello world")['input_ids']
  #[0, 31414, 232, 328, 2]
  #>>> tokenizer(" Hello world")['input_ids']
  #[0, 20920, 232, 2]

  #  https://huggingface.co/transformers/model_doc/roberta.html#robertatokenizer
  #>>> from transformers import RobertaTokenizer
  #>>> tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
  #>>> tokenizer("Hello world")['input_ids']
  #[0, 31414, 232, 328, 2]
  #>>> tokenizer(" Hello world")['input_ids']
  #[0, 20920, 232, 2]
def get_tokenizer():
  tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
  # PreTrainedTokenizer(name_or_path='roberta-base', vocab_size=50265, model_max_len=512, is_fast=False, padding_side='right', special_tokens={'bos_token': AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'eos_token': AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'unk_token': AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'sep_token': AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'pad_token': AddedToken("<pad>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'cls_token': AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'mask_token': AddedToken("<mask>", rstrip=False, lstrip=True, single_word=False, normalized=True)})
  return tokenizer
  



### (C) Dataset
class GoodreadsDataset(Dataset):

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
      'targets': torch.tensor(target, dtype=torch.float)  ## dropping "dtype=torch.long" to do linear regression
    }

###

def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = GoodreadsDataset(
    #sentences=df.summary.to_numpy(),
    #sentences=df.combined.to_numpy(),                  ## df "combined" column is "summary"+"sentenceText"
    sentences=df.sentence.to_numpy(),

    #targets=df.overall.to_numpy().astype(int),      ## type "int" causes assertion error during backprop/eval(?) because nn.output layer is type "long", not "int";
    #targets=df.overall.to_numpy().astype('int64'),  ## assertion error as well
    targets=df.has_spoiler.to_numpy(),                   ## if linear regression, then just 1 target, not 5
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=0 ## tune
    #num_workers=2  ## tune, doesn't work
  )


###
class SpoilerClassifier(nn.Module):

  def __init__(self):
    super(SpoilerClassifier, self).__init__()
    self.roberta = RobertaModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.roberta.config.hidden_size, 1)

    ## you don't do sigmoid/softmax here -- PyTorch  runs grad on the logit (more numerically stable)
    ## you only do sigmoid [torch.sigmoid(model_output)] during inference
    ## https://discuss.pytorch.org/t/using-sigmoid-output-with-cross-entropy-loss/96439/2
    ## Use BCEWithLogitsLoss  (similar to nn.crossentropyloss())
    ## do not use BCELoss (similar to nn.NLLLoss()), as it requires you to sigmoid the model's output during training, which is unnecessary (slow);  
    ##                     https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html

  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.roberta(
      input_ids=input_ids,
      attention_mask=attention_mask
    )                                   ## output.shape = pooled_output.shape = [batch,768];
    output = self.drop(pooled_output)   ## self.drop refers to the Dropout layer defined above
    return self.out(output)             ## self.out refers to the output layer defined above

###

def runEpoch(model, data_loader, loss_fn, optimizer, scheduler, n_examples):

  model = model.train()
  losses = []
  correct_predictions = 0

  for d in data_loader:
  #for d in tqdm(data_loader):
    input_ids = d["input_ids"]
    attention_mask = d["attention_mask"]
    targets = d["targets"]
    targets = targets.view(-1,1)  ## targets.shape (16) need to match outputs.shape (16,1) below; otherwise loss_fn complains below

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)  ## outputs.shape = [batch, 1] = [16,1]
    #outputs = [[0.8877], [-3.2233], [2.3421], ....]
    preds = torch.zeros_like(outputs)
    ones = torch.ones_like(preds)
    preds = torch.where(outputs < 0, preds, ones)
    #_, preds = torch.max(outputs, dim=1)  ## preds.shape = (1,16)
    ## DEBUG: DOES THE ABOVE LINE WORK AS INTENDED???????

    loss = loss_fn(outputs, targets)      ## outputs.shape = (8,1);  targets.shape = (8);  need to fix target size

    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
  ## end for each batch
  
  #return np.mean(losses)
  return correct_predictions.double() / n_examples, np.mean(losses)
## end each epoch - runEpoch()



### EVAL MODEL
def validate(model, data_loader, loss_fn, n_examples):

  model = model.eval()
  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in data_loader:
    #for d in tqdm(data_loader):
      input_ids = d["input_ids"]
      attention_mask = d["attention_mask"]
      targets = d["targets"]
      targets = targets.view(-1,1)  ## targets.shape (16) need to match outputs.shape (16,1) below; otherwise loss_fn complains below

      outputs = model(input_ids=input_ids, attention_mask=attention_mask)

      preds = torch.zeros_like(outputs)
      ones = torch.ones_like(preds)
      preds = torch.where(outputs < 0, preds, ones)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())
    ## end for each batch
  ## torch.no_grad()

  #return np.mean(losses)
  return correct_predictions.double() / n_examples, np.mean(losses)
## end validate()



######## MAIN #############
def main():

  ## load_data from json file
  df = load_data()
  #df = load_data_DEBUG()

  # split train, val
  df_train, df_val = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)

  # tokenizer
  #mytokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
  mytokenizer = get_tokenizer()



  # PyTorch data loader
  train_data_loader = create_data_loader(df_train, mytokenizer, MAX_LEN, BATCH_SIZE)
  val_data_loader = create_data_loader(df_val, mytokenizer, MAX_LEN, BATCH_SIZE)

  ### PyTorch nn
  #model = RatingRegressor()
  model = SpoilerClassifier()

  
  EPOCHS = 10

  optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
  total_steps = len(train_data_loader) * EPOCHS

  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

  #loss_fn = nn.MSELoss()
  #loss_fn = nn.CrossEntropyLoss()
  loss_fn = nn.BCEWithLogitsLoss()


  tboard = defaultdict(list)
  best_accuracy = 0
  #lowest_loss = 999

  for epoch in range(EPOCHS):

    LOG.debug(f'Epoch {epoch + 1}/{EPOCHS}')

    train_acc, train_loss = runEpoch(model, train_data_loader, loss_fn, optimizer, scheduler, len(df_train))

    LOG.debug(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = validate(model, val_data_loader, loss_fn, len(df_val))

    LOG.debug(f'Val   loss {val_loss} accuracy {val_acc}')
    
    tboard['train_acc'].append(train_acc)
    tboard['train_loss'].append(train_loss)
    tboard['val_acc'].append(val_acc)
    tboard['val_loss'].append(val_loss)

    ## SAVE TORCH MODEL IF BEST ACCURACY, BASED ON CV SET ##
    #if val_loss < lowest_loss:
    if val_acc > best_accuracy:
      torch.save(model.state_dict(), BEST_MODEL_BIN)
      #lowest_loss = val_loss
      best_accuracy = val_acc
    
  ## end for each epoch

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
  #find_maxlen_of_all_review_text()
  #find_maxlen_of_all_review_summary()
  #find_star_rating_distribution()
