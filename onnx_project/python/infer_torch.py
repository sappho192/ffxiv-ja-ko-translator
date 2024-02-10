import transformers
import torch

encoder_model_name = "cl-tohoku/bert-base-japanese-v2"
decoder_model_name = "skt/kogpt2-base-v2"
src_tokenizer = transformers.BertJapaneseTokenizer.from_pretrained(encoder_model_name)
trg_tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(decoder_model_name)
model = transformers.EncoderDecoderModel.from_pretrained("sappho192/ffxiv-ja-ko-translator")


def translate(text_src):
    embeddings = src_tokenizer(text_src, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
    embeddings = {k: v for k, v in embeddings.items()}
    output = model.generate(**embeddings,
                            num_beams=5, num_return_sequences=1, no_repeat_ngram_size=1, remove_invalid_values=True)[0, 1:-1]
    text_trg = trg_tokenizer.decode(output.cpu())
    return text_trg

text = "初めまして。"
print(translate(text))

texts = [
    "逃げろ!",  # Should be "도망쳐!"
    "初めまして.",  # "반가워요"
    "よろしくお願いします.",  # "잘 부탁드립니다."
    "ギルガメッシュ討伐戦",  # "길가메쉬 토벌전"
    "ギルガメッシュ討伐戦に行ってきます。一緒に行きましょうか？",  # "길가메쉬 토벌전에 갑니다. 같이 가실래요?"
    "夜になりました",  # "밤이 되었습니다"
    "ご飯を食べましょう."  # "음, 이제 식사도 해볼까요"
 ]

for text in texts:
    print(translate(text))
    print()

#
# sm = torch.jit.script(model)
# sm.save("torch_model_script.pt")
