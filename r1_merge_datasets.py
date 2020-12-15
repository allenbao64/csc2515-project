import gzip
import json
import pandas as pd
import pickle

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


from r0_constants import FILE0_GOODREADS_BOOKS
from r0_constants import FILE1_BOOK_ID_TITLE
from r0_constants import FILE2_SPOILER_ALLEN
from r0_constants import FILE3_FLATTENED

from r0_constants import N_TOTAL_SENTENCES


# EXTRACT ONLY "book_id" and "title" from 2GB 'goodreads_books.json.gz'
def extract_only_book_id_and_title_PythonDict(infile, outfile, head = None):
  count = 0
  my_dict={}
  with gzip.open(infile) as fin:
    for l in fin:
      d = json.loads(l)

      my_dict[d.get('book_id')] = d.get('title')
      count += 1

      if (count % 100000) == 0:
        LOG.debug("{}) book_id: {}; title:{}".format(count, d.get('book_id'), d.get('title')))
      
      # break if reaches the head-th line
      if (head is not None) and (count >= head):
        break
    ## end for
  ## end with infile

  with gzip.open(outfile, 'wb') as fout:
    pickle.dump(my_dict, fout, protocol=pickle.HIGHEST_PROTOCOL)

## end extract_only_book_id_and_title_PythonDict()



###
''' SAMPLE PROPOSED FLATTENED DATA
{'user_id': '01ec1a320ffded6b2dd47833f2c8e4fb',                                                               
 'timestamp': '2013-12-28',                                                                                   
 'book_id': '18398089',                                                                                       
 'book_title': 'Harry Potter and the Deathly Hallows'
 'sent_num': 0
 'sent_spoil': 0
 'sentence': 'This book started in a pretty bland way.'
 'rating': 5,                                                                                                 
 'has_spoiler': False,                                                                                        
 'review_id': '4b3ffeaf14310ac6854f140188e191cd'}   

'''
def create_flatten_version(spoiler_file, book_title_file, out_file, head = None):
  #open_allen_file()
  spoiler_df = pd.read_pickle(spoiler_file)   #  275607 reviews

  # open book-title file
  title_dict = None

  with gzip.open(book_title_file, 'rb') as f_bt:
    title_dict = pickle.load(f_bt)             # 2360655 books
  #OBSOLETE: title_df = pd.read_pickle(book_title_file)  # 2360655 books

  out_rows = []   ## each item in this list is "a row of review-sentence-data"
  count = 0
  ## saarthak's code
  for i in range(len(spoiler_df)):  # for each review

    this_review = spoiler_df.iloc[i]
    book_id = this_review['book_id']

    book_title = title_dict[book_id]
    #book_title = title_df.loc[(title_df['book_id'] == book_id)].iloc[0].at['title']
    if not book_title:   #if (book_title is None) or (book_title == '') or (len(book_title) == 0):
      LOG.debug("book id {} has no title".format(book_id))

    sentences = this_review['review_sentences']

    for j in range(len(sentences)):  # for each sentence in that review
      tempList = []
      tempList.append(this_review['user_id'])
      tempList.append(this_review['timestamp'])
      tempList.append(book_id)
      tempList.append(book_title)
      tempList.append(j)  # sent_num
      tempList.append(sentences[j][0])  # sent_spoil
      tempList.append(sentences[j][1])  # sentence
      tempList.append(this_review['rating'])
      tempList.append(this_review['has_spoiler'])
      tempList.append(this_review['review_id'])

      count += 1
      if (count % 1000) == 0:
        LOG.debug(tempList)

      out_rows.append(tempList)
    ## end for j

    # break if reaches the head-th line
    if (head is not None) and (count >= head):
      break   ## break out of loop i
  ## end for i

  flattened_df = pd.DataFrame(out_rows, columns=['user_id','timestamp','book_id','book_title','sent_num','sent_spoil','sentence','rating','has_spoiler','review_id'])
  with gzip.open(out_file, 'wb') as fout:
    pickle.dump(flattened_df, fout)   #protocol=pickle.HIGHEST_PROTOCOL

## end create_flatten_version()



###
def main():
  # UNCOMMENT 1 OF THE LINES BELOW

  # 1) extract book id and title from 2GB file 
  #extract_only_book_id_and_title_PythonDict(FILE0_GOODREADS_BOOKS, FILE1_BOOK_ID_TITLE, 500)  # debug: only 500 lines
  #extract_only_book_id_and_title_PythonDict(FILE0_GOODREADS_BOOKS, FILE1_BOOK_ID_TITLE)

  # 2) open Allen's pkl and look up book title for each review's book_id, and spit out a flattened version
  #create_flatten_version(FILE2_SPOILER_ALLEN, FILE1_BOOK_ID_TITLE, FILE3_FLATTENED, 10000)  # debug: only 500 lines
  create_flatten_version(FILE2_SPOILER_ALLEN, FILE1_BOOK_ID_TITLE, FILE3_FLATTENED, N_TOTAL_SENTENCES)

  pass
## end main()


if __name__ == '__main__':
  main()
