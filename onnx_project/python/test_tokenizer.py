from transformers import BertJapaneseTokenizer, PreTrainedTokenizerFast

encoder_model_name = "cl-tohoku/bert-base-japanese-v2"
decoder_model_name = "skt/kogpt2-base-v2"

src_tokenizer = BertJapaneseTokenizer.from_pretrained(encoder_model_name)

text = "こんにちは、私は日本語を話せます。"
encoded_input = src_tokenizer.encode_plus(text)
print(f'encoded_input: {encoded_input}')
decoded_input = src_tokenizer.decode(encoded_input['input_ids'], skip_special_tokens=True)
print(f'decoded_input: {decoded_input}')
