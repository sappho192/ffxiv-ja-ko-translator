from transformers import(
    EncoderDecoderModel,
    PreTrainedTokenizerFast,
    # XLMRobertaTokenizerFast,
    BertJapaneseTokenizer,
    BertTokenizerFast,
)

import pandas as pd
csv_test = pd.read_csv('./output/ffac_full.csv')
# csv_test = pd.read_csv('ffac_test.csv')

import csv

encoder_model_name = "cl-tohoku/bert-base-japanese-v2"
decoder_model_name = "skt/kogpt2-base-v2"

src_tokenizer = BertJapaneseTokenizer.from_pretrained(encoder_model_name)
trg_tokenizer = PreTrainedTokenizerFast.from_pretrained(decoder_model_name)
model = EncoderDecoderModel.from_pretrained("./dump/best_model")

def main():
    data_test = []
    data_test_label = []
    data_test_infer = []
    for row in csv_test.itertuples():
        data_test.append(row[1])
        data_test_label.append(row[2])

    for text in data_test:
        embeddings = src_tokenizer(text, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
        embeddings = {k: v for k, v in embeddings.items()}
        output = model.generate(**embeddings)[0, 1:-1]
        result = trg_tokenizer.decode(output.cpu())
        # print(result)
        data_test_infer.append(result)
    
    rows = zip(data_test, data_test_infer, data_test_label)
    with open('test_result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'inference', 'answer'])
        for row in rows:
            writer.writerow(row)

if __name__ == "__main__":
    main()