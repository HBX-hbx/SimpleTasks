import random
import argparse
from pathlib import Path


def split(src_fpath, tgt_fpath, new_data_dir, ratio=(0.9, 0.05, 0.05), src='en', tgt='zh'):
    new_data_dir = Path(new_data_dir)
    
    src_fp = open(src_fpath, encoding='utf-8')
    tgt_fp = open(tgt_fpath, encoding='utf-8')
    
    src_train_fp = open(new_data_dir.joinpath('train.' + src), 'w', encoding='utf-8')
    src_val_fp = open(new_data_dir.joinpath('val.' + src), 'w', encoding='utf-8')
    src_test_fp = open(new_data_dir.joinpath('test.' + src), 'w', encoding='utf-8')
    
    tgt_train_fp = open(new_data_dir.joinpath('train.' + tgt), 'w', encoding='utf-8')
    tgt_val_fp = open(new_data_dir.joinpath('val.' + tgt), 'w', encoding='utf-8')
    tgt_test_fp = open(new_data_dir.joinpath('test.' + tgt), 'w', encoding='utf-8')
    
    for s, t in zip(src_fp.readlines(), tgt_fp.readlines()):
        rand = random.random()
        if 0 < rand <= ratio[0]:
            src_train_fp.write(s)
            tgt_train_fp.write(t)
        elif ratio[0] < rand <= ratio[0] + ratio[1]:
            src_val_fp.write(s)
            tgt_val_fp.write(t)
        else:
            src_test_fp.write(s)
            tgt_test_fp.write(t)
    
    src_fp.close()
    tgt_fp.close()
    src_train_fp.close()
    tgt_train_fp.close()
    src_val_fp.close()
    tgt_val_fp.close()
    src_test_fp.close()
    tgt_test_fp.close()


parser = argparse.ArgumentParser()
parser.add_argument('--src_fpath', type=str)
parser.add_argument('--tgt_fpath', type=str)
parser.add_argument('--new_data_dir', type=str)
args = parser.parse_args()

split(args.src_fpath, args.tgt_fpath, args.new_data_dir)