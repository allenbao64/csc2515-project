from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

#from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup  #, BertConfig
from transformers import AutoTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup

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
from r0_constants import FILE4_FLATTENED_TRAIN
from r0_constants import FILE5_FLATTENED_CV

from r0_constants import N_TOTAL_SENTENCES
from r0_constants import COLUMNS_FLATTENED

PRE_TRAINED_MODEL_NAME = 'roberta-base'
BEST_MODEL_BIN = 'best_model_state.bin'
MAX_LEN = 40     ## tune
BATCH_SIZE = 16  ## tune
EPOCHS = 5       ## tune




### load data
def load_data(file):
  df = pd.read_pickle(file)

  cols = ['book_title', 'sentence']
  df['combined'] = df[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)  # do not put this line in a for loop (that will be O(N^2))

  LOG.debug("num_samples: {}".format(df.shape[0]))
  return df



###
def load_data_DEBUG(file):
  df = pd.read_pickle(file)

  df2 = df.head(100)

  LOG.debug("########## YOU ONLY HAVE {} SAMPLES !!! ##########".format(df2.shape[0]))
  LOG.debug("########## YOU ONLY HAVE {} SAMPLES !!! ##########".format(df2.shape[0]))
  LOG.debug("num_samples: {}".format(df2.shape[0]))
  LOG.debug("########## YOU ONLY HAVE {} SAMPLES !!! ##########".format(df2.shape[0]))
  LOG.debug("########## YOU ONLY HAVE {} SAMPLES !!! ##########".format(df2.shape[0]))

  cols = ['book_title', 'sentence']
  df2['combined'] = df2[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)  # do not put this line in a for loop (that will be O(N^2))

  return df2



###
def get_tokenizer():
  tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
  # PreTrainedTokenizer(name_or_path='roberta-base', vocab_size=50265, model_max_len=512, is_fast=False, padding_side='right', special_tokens={'bos_token': AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'eos_token': AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'unk_token': AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'sep_token': AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'pad_token': AddedToken("<pad>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'cls_token': AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 'mask_token': AddedToken("<mask>", rstrip=False, lstrip=True, single_word=False, normalized=True)})
  return tokenizer



### Dataset
class FlattenedDataset(Dataset):

  def __init__(self, combineds, targets, tokenizer, max_len):
    self.combineds = combineds
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.combineds)
  
  def __getitem__(self, item):
    combined = str(self.combineds[item])
    target = self.targets[item]

    encoding = self.tokenizer.encode_plus(
      combined,
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
      'combined': combined,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.float)
    }

## end class FlattenedDataset


###
def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = FlattenedDataset(
    combineds=df.combined.to_numpy(),                  ## df "combined" column is "title"+"sentence"
    targets=df.sent_spoil.to_numpy(),
    #targets=df.sent_spoil.astype(float).to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=0    ## tune
  )
## end create_data_loader()



###
class SpoilerClassifier(nn.Module):

  def __init__(self):
    super(SpoilerClassifier, self).__init__()
    self.roberta = RobertaModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.roberta.config.hidden_size, 1)
    self.out_act = nn.Sigmoid()
        
    ## you don't do sigmoid/softmax here -- PyTorch  runs grad on the logit (more numerically stable)
    ## you only do sigmoid [torch.sigmoid(model_output)] during inference
    ## https://discuss.pytorch.org/t/using-sigmoid-output-with-cross-entropy-loss/96439/2
    ## Use BCEWithLogitsLoss  (similar to nn.crossentropyloss())
    ## do not use BCELoss (similar to nn.NLLLoss()), as it requires you to sigmoid the model's output during training, which is unnecessary (slow);  
    ##                     https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html

  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)  ## output.shape = pooled_output.shape = [batch,768];
    output = self.drop(pooled_output)   ## self.drop refers to the Dropout layer defined above
    z2 = self.out(output)
    a2 = self.out_act(z2)
    return a2


    #return self.out(output)             ## self.out refers to the output layer defined above

## end class SpoilerClassifier



def runEpoch(model, data_loader, loss_fn, optimizer, scheduler, n_examples):

  model = model.train()
  
  #losses = []
  #correct_predictions = 0

  y_hats = []    # prediction
  y_labels = []  # ground truth

  #for d in tqdm(data_loader):
  for d in data_loader:
    input_ids = d["input_ids"]
    attention_mask = d["attention_mask"]
    targets = d["targets"]
    targets = targets.view(-1,1)  ## targets.shape (16) need to match outputs.shape (16,1) below; otherwise loss_fn complains below

    optimizer.zero_grad()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)  ## outputs.shape = [batch, 1] = [16,1]
    #outputs = [[0.8877], [-3.2233], [2.3421], ....]
    preds = torch.zeros_like(outputs)
    ones = torch.ones_like(preds)
    #preds = torch.where(outputs < 0, preds, ones)  ## logit
    preds = torch.where(outputs < 0.5, preds, ones)  ## sigmoid
    ## DEBUG: DOES THE ABOVE LINE WORK AS INTENDED??????? [yes]

    loss = loss_fn(outputs, targets)      ## outputs.shape = (8,1);  targets.shape = (8);  need to fix target size

    # collect stats for this batch
    #correct_predictions += torch.sum(preds == targets)
    #losses.append(loss.item())
    y_hats.extend(preds.flatten().tolist())
    y_labels.extend(targets.flatten().tolist())

    # backprop, take gradient step
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    #optimizer.zero_grad()
  ## end for each batch
  
  #return np.mean(losses)
  #return correct_predictions.double() / n_examples, np.mean(losses)
  return y_hats, y_labels
## end each epoch - runEpoch()



### EVAL MODEL
def validate(model, data_loader, loss_fn, n_examples):

  model = model.eval()
  
  #losses = []
  #correct_predictions = 0
  
  y_hats = []    # prediction
  y_labels = []  # ground truth

  with torch.no_grad():
    #for d in tqdm(data_loader):
    for d in data_loader:
      input_ids = d["input_ids"]
      attention_mask = d["attention_mask"]
      targets = d["targets"]
      targets = targets.view(-1,1)  ## targets.shape (16) need to match outputs.shape (16,1) below; otherwise loss_fn complains below

      outputs = model(input_ids=input_ids, attention_mask=attention_mask)

      preds = torch.zeros_like(outputs)
      ones = torch.ones_like(preds)
      #preds = torch.where(outputs < 0, preds, ones)  ## logit
      preds = torch.where(outputs < 0.5, preds, ones)  ## sigmoid

      # collect stats for this batch
      #loss = loss_fn(outputs, targets)

      #correct_predictions += torch.sum(preds == targets)
      #losses.append(loss.item())
      y_hats.extend(preds.flatten().tolist())
      y_labels.extend(targets.flatten().tolist())
    ## end for each batch
  ## torch.no_grad()

  #return np.mean(losses)
  #return correct_predictions.double() / n_examples, np.mean(losses)
  return y_hats, y_labels
## end validate()



######## MAIN #############
def main():

  ## load training data
  df_train = load_data(FILE4_FLATTENED_TRAIN)
  #df_train = load_data_DEBUG(FILE4_FLATTENED_TRAIN)

  ## load validation data
  df_val = load_data(FILE5_FLATTENED_CV)
  
  # tokenizer
  mytokenizer = get_tokenizer()

  # PyTorch data loader
  train_data_loader = create_data_loader(df_train, mytokenizer, MAX_LEN, BATCH_SIZE)
  val_data_loader = create_data_loader(df_val, mytokenizer, MAX_LEN, BATCH_SIZE)

  ### PyTorch nn
  model = SpoilerClassifier()
  optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
  total_steps = len(train_data_loader) * EPOCHS

  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

  #loss_fn = nn.MSELoss()           # linear regression
  #loss_fn = nn.CrossEntropyLoss()  # multi-class classification
  loss_fn = nn.BCELoss()
  #loss_fn = nn.BCEWithLogitsLoss()  # binary classification

  #history = defaultdict(list)
  #best_accuracy = 0
  best_auc_score = 0

  for epoch in range(EPOCHS):
    LOG.debug(f'Epoch {epoch + 1}/{EPOCHS}')

    #train_acc, train_loss = runEpoch(model, train_data_loader, loss_fn, optimizer, scheduler, len(df_train))
    #LOG.debug(f'Train loss {train_loss} accuracy {train_acc}')
    train_y_hats, train_y_labels = runEpoch(model, train_data_loader, loss_fn, optimizer, scheduler, len(df_train))
    #print(classification_report(train_y_labels, train_y_hats))
    LOG.debug("TRAIN - CLASSIFICATION REPORT\n{}".format(classification_report(train_y_labels, train_y_hats, target_names=['no_spoilers', 'has_spoilers'])))
    train_auc_score = roc_auc_score(train_y_labels, train_y_hats)
    LOG.debug("train_auc_score: {}".format(train_auc_score))

    #val_acc, val_loss = validate(model, val_data_loader, loss_fn, len(df_val))
    #LOG.debug(f'Val   loss {val_loss} accuracy {val_acc}')
    val_y_hats, val_y_labels = validate(model, val_data_loader, loss_fn, len(df_val))
    LOG.debug("VAL - CLASSIFICATION REPORT\n{}".format(classification_report(val_y_labels, val_y_hats)))  ## 3rd param: target_names=class_names
    val_auc_score = roc_auc_score(val_y_labels, val_y_hats)
    LOG.debug("val_auc_score: {}".format(val_auc_score))

    #history['train_acc'].append(train_acc)
    #history['train_loss'].append(train_loss)
    #history['val_acc'].append(val_acc)
    #history['val_loss'].append(val_loss)

    ## SAVE TORCH MODEL IF BEST ACCURACY, BASED ON CV SET ##
    #if val_acc > best_accuracy:
    #  torch.save(model.state_dict(), BEST_MODEL_BIN)
    #  best_accuracy = val_acc
    if val_auc_score > best_auc_score:
      torch.save(model.state_dict(), BEST_MODEL_BIN)
      best_auc_score = val_auc_score
    
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
