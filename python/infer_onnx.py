from transformers import BertJapaneseTokenizer,PreTrainedTokenizerFast
from optimum.onnxruntime import ORTModelForSeq2SeqLM
import torch

encoder_model_name = "cl-tohoku/bert-base-japanese-v2"
decoder_model_name = "skt/kogpt2-base-v2"

src_tokenizer = BertJapaneseTokenizer.from_pretrained(encoder_model_name)
trg_tokenizer = PreTrainedTokenizerFast.from_pretrained(decoder_model_name)

# `from_transformers=True` downloads the PyTorch weights and converts them to ONNX format
model = ORTModelForSeq2SeqLM.from_pretrained("./onnx")
text = "ギルガメッシュ討伐戦"
# text2 = "ギルガメッシュ討伐戦に行ってきます。一緒に行きましょうか？"
text2 = "ご飯を食べましょう."
texts = [
    "逃げろ!",  # Should be "도망쳐!"
    "初めまして.",  # "반가워요"
    "よろしくお願いします.",  # "잘 부탁드립니다."
    "ギルガメッシュ討伐戦",  # "길가메쉬 토벌전"
    "ギルガメッシュ討伐戦に行ってきます。一緒に行きましょうか？",  # "길가메쉬 토벌전에 갑니다. 같이 가실래요?"
    "夜になりました",  # "밤이 되었습니다"
    "ご飯を食べましょう."  # "음, 이제 식사도 해볼까요"
 ]


def translate(text_src):
    embeddings = src_tokenizer(text_src, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
    embeddings = {k: v for k, v in embeddings.items()}

    output = model.generate(**embeddings)[0, 1:-1]
    text_trg = trg_tokenizer.decode(output.cpu())
    return text_trg


# print(translate(text))
# print(translate(text2))
for text in texts:
    print(translate(text))
