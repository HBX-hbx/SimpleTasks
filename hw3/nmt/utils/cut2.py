import argparse
from pathlib import Path

def cut2(fpath, src_tgt_dir, src='en', tgt='zh'):
    src_tgt_dir = Path(src_tgt_dir)
    fp = open(fpath, encoding='utf-8')
    src_fp = open(src_tgt_dir.joinpath('raw.' + src), 'w', encoding='utf-8')
    tgt_fp = open(src_tgt_dir.joinpath('raw.' + tgt), 'w', encoding='utf-8')
    for line in fp.readlines():
        src_line, tgt_line = line.replace('\n', '').split('\t')
        src_fp.write(src_line + '\n')
        tgt_fp.write(tgt_line + '\n')
    fp.close()
    src_fp.close()
    tgt_fp.close()

parser = argparse.ArgumentParser()
parser.add_argument('--fpath', type=str)
parser.add_argument('--src_tgt_dir', type=str)
parser.add_argument('--src', type=str, default='en')
parser.add_argument('--tgt', type=str, default='zh')
args = parser.parse_args()

cut2(fpath=args.fpath, src_tgt_dir=args.src_tgt_dir, src=args.src, tgt=args.tgt)