import torch
import pickle
import numpy as np
from typing import List
from transformers import PegasusForConditionalGeneration, PegasusTokenizer,AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_splitter import SentenceSplitter, split_text_into_sentences
splitter = SentenceSplitter(language='en')
import warnings
warnings.filterwarnings("ignore")


torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

first_model=pickle.load(open("first_model","rb")).to(torch_device)
first_tokenizer=pickle.load(open("first_tokenizer","rb"))


second_model=pickle.load(open("second_model","rb")).to(torch_device)
#second_tokenizer=pickle.load(open("second_tokenizer","rb"))
model_name = 'tuner007/pegasus_paraphrase' 
second_tokenizer = PegasusTokenizer.from_pretrained(model_name)





  
  
def generate_text(inp):
    context = inp
    text = "paraphrase: "+context + " </s>"
    encoding = first_tokenizer.encode_plus(text,max_length =256, truncation=True, return_tensors="pt")
    input_ids,attention_mask  = encoding["input_ids"].to(torch_device), encoding["attention_mask"].to(torch_device)
    first_model.eval()
    diverse_beam_outputs = first_model.generate(
    input_ids=input_ids,attention_mask=attention_mask,
    max_length=256,
    early_stopping=True,
    num_beams=5,
    num_beam_groups = 5,
    num_return_sequences=5,
    diversity_penalty = 0.90)
    sent = first_tokenizer.decode(diverse_beam_outputs[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)


def get_response(input_text: str, num_return_sequences: int, num_beams: int) -> List[str]:
      batch = second_tokenizer([input_text], truncation = True, padding = 'longest', max_length = 60,return_tensors="pt").to(torch_device)
      translated = second_model.generate(**batch, max_length = 60, num_beams = num_beams, num_return_sequences = num_return_sequences, temperature = 1.5,output_scores=True)
      tgt_text = second_tokenizer.batch_decode(translated, skip_special_tokens = True)
      return tgt_text
    

    
    
def paraphraser(inp_text):
  sentence_list = splitter.split(inp_text)
  paraphrased=[]
  for sent in sentence_list:
    if len(sent.split())<28:
      outputs = get_response(sent,10,10)
      paraphrased.append(outputs[np.argmax([len(result) for result in outputs])])
      
    elif len(sent.split())>28 and len(sent.split())<32:
      paraphrased.append(generate_text(sent))
    else:
      splited_sent=[]
      splitted=sent.split(',')
      for i in range(len(splitted)):
        if i<len(splitted)-1:
          joined = ",".join(splitted[i:i+2])
          if len(joined.split())<25:
            splitted[i] = joined
            del splitted[i+1]
      for s_sent in splitted:
        if len(s_sent.split())<20:
          outputs = get_response(s_sent,10,10)
          splited_sent.append(outputs[np.argmax([len(result) for result in outputs])])
        else:
          splited_sent.append(generate_text(s_sent))
      paraphrased.append("".join(splited_sent))
  paraphrased = " ".join(paraphrased)
  return paraphrased

    
    
    
