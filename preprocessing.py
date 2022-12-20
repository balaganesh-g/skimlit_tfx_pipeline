
import os
import tempfile
import pandas as pd

try:
    _data_root = os.mkdir('tfx-data')
except FileExistsError:
   # directory already exists
   pass

_data_root = os.getcwd()+'/tfx-data'
data_dir = '/content/pubmed-rct/PubMed_20k_RCT_numbers_replaced_with_at_sign'

def read_lines(filename):
    """args:
            filename - name/path of the file 
       returns:
            reads line by and make each sentence as separate line and return a list of strings
    """
    with open(filename, 'r') as f:
        return f.readlines()

def preprocess_text_with_line_numbers(filename):
    train_eg = read_lines(filename)
    abstract_samples = []
    abstract_lines = ""
    for line in train_eg:
        if line.startswith('###'):
            abstract_id = line
            abstract_lines = ""
        elif line.isspace():
            abstract_line_split = abstract_lines.splitlines()
            
            for abstract_line_number,abstract_line in enumerate(abstract_line_split):
                line_data = {}
                target_line_and_label = abstract_line.split('\t')
                line_data['target'] = target_line_and_label[0]
                line_data['text'] = target_line_and_label[1] 
                line_data['line_number'] = abstract_line_number
                line_data['total_lines'] = len(abstract_line_split)
                abstract_samples.append(line_data)
        else:
            abstract_lines += line 
    return abstract_samples     

train_samples = preprocess_text_with_line_numbers(data_dir +'/'+ "train.txt")
val_sample = preprocess_text_with_line_numbers(data_dir+'/'+ "dev.txt")
test_sample = preprocess_text_with_line_numbers(data_dir+'/'+ "test.txt")


train_df = pd.DataFrame(train_samples)
val_df = pd.DataFrame(val_sample)
test_df = pd.DataFrame(test_sample)



train_df['target_int'] = pd.Categorical(train_df['target']).codes
val_df['target_int'] = pd.Categorical(val_df['target']).codes
test_df['target_int'] = pd.Categorical(test_df['target']).codes


train_df = train_df.to_csv(os.path.join(_data_root,"train.csv"))
val_df = val_df.to_csv(os.path.join(_data_root,"val.csv"))
test_df = test_df.to_csv(os.path.join(_data_root,"test.csv"))
