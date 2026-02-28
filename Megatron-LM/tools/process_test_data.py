import os
import jsonlines
import tqdm
from multiprocessing import Pool
import sys
import transformers

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from megatron.core.datasets import indexed_dataset

POOLNUM = 64
MAX_LENGTH = 4096
VOCAB_SIZE = 131072
EOD = 2
ORIGIN_DATA_PATH = "/mnt/yanguo.sun/dianxin/full_data_40"
BIN_DATA_PATH = "/mnt/yanguo.sun/dianxin/bin_data_40"
TOKENIZER_PATH = "/mnt/yanguo.sun/dianxin/TeleChat3-36B-Thinking"


def get_datas(file):
    datas = []
    tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    f = jsonlines.open(os.path.join(ORIGIN_DATA_PATH, file), 'r')
    for data in tqdm.tqdm(f):
        datas.append(tokenizer(data["data"])["input_ids"])
    f.close()
    return datas


def get_tokens(datas):
    tokens = []
    inputs_ids = []
    for data in datas:
        inputs_ids += data + [EOD]
        while len(inputs_ids) >= MAX_LENGTH:
            tokens.append(
                {"input_ids": inputs_ids[:MAX_LENGTH]})
            inputs_ids = inputs_ids[MAX_LENGTH:]
    return tokens


def run(file):
    datas = get_datas(file)
    print(f'!!!!!! len of datas is {len(datas)} !!!!!')
    if len(datas) < 1: return
    tokens = get_tokens(datas)
    filename = os.path.splitext(file)[0]
    print(len(tokens))
    builder = indexed_dataset.IndexedDatasetBuilder(
        os.path.join(BIN_DATA_PATH, "test_"+filename) + ".bin",
        dtype=indexed_dataset.DType.optimal_dtype(VOCAB_SIZE),
    )
    for doc in tokens:
        builder.add_document(doc["input_ids"], [MAX_LENGTH])
    builder.finalize(os.path.join(BIN_DATA_PATH,"test_"+filename) + ".idx")
    print(f"{file} finish\n")


if __name__ == '__main__':
    os.mkdir(BIN_DATA_PATH)
    files = os.listdir(ORIGIN_DATA_PATH)
    print(files)
    pool = Pool(POOLNUM)
    results = []
    for file in files:
        # run(file)
        pool.apply_async(run, args=(file,))
    pool.close()
    pool.join()
