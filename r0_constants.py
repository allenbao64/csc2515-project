RANDOM_SEED = 1234

FILE0_GOODREADS_BOOKS = './goodreads_books.json.gz'   # 2GB dataset with book title
FILE1_BOOK_ID_TITLE = './book_id_title.pkl.gz'        # Marshall extracted (book_id,book_title) from above 2GB file
FILE2_SPOILER_ALLEN = './goodreads.pkl.gz'            # Allen's shrinked-down spoiler dataset
FILE3_FLATTENED = './flattened.pkl.gz'                # 1M samples
#FILE3_FLATTENED = './flattened_300K.pkl.gz'          #  300K samples
#FILE3_FLATTENED = './flattened_10K.pkl.gz'           #   10K samples

#FILE4_FLATTENED_TRAIN = './flattened_train.pkl.gz'  # https://drive.google.com/file/d/17SmffPfmeVMICKjGmcbgODvH5cypX3gB/view?usp=sharing
#FILE5_FLATTENED_CV = './flattened_cv.pkl.gz'        # https://drive.google.com/file/d/1vrAPRUJ-8a-W-3-WPfm2sdzdS5w0BcK3/view?usp=sharing
#FILE6_FLATTENED_TEST = './flattened_test.pkl.gz'    # https://drive.google.com/file/d/1SLlgeOQGJ-kd8v4foZ_TH5IX12_4wcy4/view?usp=sharing
#N_TOTAL_SENTENCES = 1000000

FILE4_FLATTENED_TRAIN = './flattened_270K_train.pkl.gz'
FILE5_FLATTENED_CV = './flattened_15K_cv.pkl.gz'
FILE6_FLATTENED_TEST = './flattened_15K_test.pkl.gz'
N_TOTAL_SENTENCES = 300000

#FILE4_FLATTENED_TRAIN = './flattened_9K_train.pkl.gz'
#FILE5_FLATTENED_CV = './flattened_0.5K_cv.pkl.gz'
#FILE6_FLATTENED_TEST = './flattened_0.5K_test.pkl.gz'
#N_TOTAL_SENTENCES =    10000


COLUMNS_FLATTENED = ['user_id','timestamp','book_id','book_title','sent_num','sent_spoil','sentence','rating','has_spoiler','review_id']
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
