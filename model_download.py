import torch
import numpy as np
from typing import List
import pickle
from transformers import PegasusForConditionalGeneration, PegasusTokenizer,AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_splitter import SentenceSplitter, split_text_into_sentences
splitter = SentenceSplitter(language='en')
import warnings
warnings.filterwarnings("ignore")



torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""T5 Model"""
first_model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality").to(torch_device)
first_tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")

"""Pegasus Model"""
model_name = 'tuner007/pegasus_paraphrase' 
second_tokenizer = PegasusTokenizer.from_pretrained(model_name)
second_model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)



filename1 = 'first_model'
filename2 = 'first_tokenizer'

filename3 = 'second_model'
filename4 = 'second_tokenizer'


pickle.dump(first_model, open(filename1, 'wb'))
pickle.dump(first_tokenizer,open(filename2, 'wb'))

pickle.dump(second_model, open(filename3, 'wb'))
pickle.dump(second_tokenizer,open(filename4, 'wb'))