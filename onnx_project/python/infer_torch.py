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
    output = model.generate(**embeddings, max_length=300)[0, 1:-1]
    text_trg = trg_tokenizer.decode(output.cpu())
    return text_trg

text = "ギルガメッシュ討伐戦"
print(translate(text))
#
# sm = torch.jit.script(model)
# sm.save("torch_model_script.pt")
