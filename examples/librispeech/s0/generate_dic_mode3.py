import csv 
import sys 
import pickle
import torch
import sentencepiece as spm
from wenet.utils.file_utils import read_symbol_table
def tokenize(txt_list,
             symbol_table,
             ):
    """ Decode text to chars or BPE
        Inplace operation

        Args:
            data: Iterable[{key, wav, txt, sample_rate}]

        Returns:
            Iterable[{key, wav, txt, tokens, tokens, sample_rate}]
    """
    tokens = []
    for ch in txt_list:
        if ch=='\n' :
            continue
        elif ch in symbol_table:
            tokens.append(symbol_table[ch])
        elif '<unk>' in symbol_table:
            tokens.append(symbol_table['<unk>'])
    return tokens


if __name__ == "__main__":
    tsv_file = open("/home/work_nfs6/tyxu/workspace/wenet-rnnt-runtime/examples/librispeech/s0/data/Ch_context_list")
    reader = csv.reader(tsv_file,delimiter='\t')
    symbol = read_symbol_table("/home/work_nfs6/tyxu/workspace/wenet-rnnt-runtime/examples/aishell/rnnt/data/dict/lang_char.txt")
    dic = {}
    res=[]
    for line in tsv_file:

        # print(context_list)
        res.append(tokenize(line,symbol))
    with open(r'data/processed_context_test_ch', 'w') as fp:
        for item in res:
            # write each item on a new line
            
            fp.write("%s\n" % ' '.join(str(word) for  word in item))
    print("Total Hot Words: %2d" % (len(res)))
    print('Done')
        # import pdb;pdb.set_trace()





