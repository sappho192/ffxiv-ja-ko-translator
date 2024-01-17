from onnxruntime import InferenceSession, SessionOptions
from transformers import BertJapaneseTokenizer, PreTrainedTokenizerFast
import numpy as np
import os

encoder_model_name = "cl-tohoku/bert-base-japanese-v2"
decoder_model_name = "skt/kogpt2-base-v2"

src_tokenizer = BertJapaneseTokenizer.from_pretrained(encoder_model_name)
trg_tokenizer = PreTrainedTokenizerFast.from_pretrained(decoder_model_name)
trg_tokenizer.bos_token_id = 0
trg_tokenizer.eos_token_id = 1
trg_tokenizer.pad_token_id = 3
trg_tokenizer.unk_token_id = 5
trg_tokenizer.mask_token_id = 6


def download_model():
    from pretty_downloader import download

    # If onnx2 directory doesn't exist, create it
    if not os.path.exists('onnx'):
        os.makedirs('onnx')

    ONNX_PATH_ROOT = 'onnx'

    # Check if file already exists
    if os.path.exists(f'{ONNX_PATH_ROOT}/encoder_model.onnx'):
        print('encoder_model.onnx already exists. Skipping download.')
    else:
        download('https://huggingface.co/sappho192/ffxiv-ja-ko-translator/resolve/main/onnx/encoder_model.onnx',
                 file_path=ONNX_PATH_ROOT)

    if os.path.exists(f'{ONNX_PATH_ROOT}/decoder_model_merged.onnx'):
        print('decoder_model_merged.onnx already exists. Skipping download.')
    else:
        download('https://huggingface.co/sappho192/ffxiv-ja-ko-translator/resolve/main/onnx/decoder_model_merged.onnx',
                 file_path=ONNX_PATH_ROOT)

    if os.path.exists(f'{ONNX_PATH_ROOT}/generation_config.json'):
        print('generation_config.json already exists. Skipping download.')
    else:
        download('https://huggingface.co/sappho192/ffxiv-ja-ko-translator/resolve/main/onnx/generation_config.json',
                 file_path=ONNX_PATH_ROOT)

    if os.path.exists(f'{ONNX_PATH_ROOT}/config.json'):
        print('config.json already exists. Skipping download.')
    else:
        download('https://huggingface.co/sappho192/ffxiv-ja-ko-translator/resolve/main/onnx/config.json',
                 file_path=ONNX_PATH_ROOT)


print('Downloading onnx model and config files into onnx folder...')
download_model()
print('Model preparation complete.')

sess_options = SessionOptions()
sess_options.log_severity_level = 3 # mute warnings including CleanUnusedInitializersAndNodeArgs

decoder_session = InferenceSession("./onnx/decoder_model_merged.onnx",sess_options=sess_options)
encoder_session = InferenceSession("./onnx/encoder_model.onnx",sess_options=sess_options)


def greedy_search(_input_data, _encoder_session, _decoder_session, _trg_tokenizer, max_length=50):
    # Assuming `input_ids` is the output from the encoder session
    # Initialize the input for the decoder
    _input_data['input_ids'] = np.array([[_trg_tokenizer.bos_token_id]]).astype(np.int64)

    # Initialize the list to store the generated tokens
    generated_tokens = []

    # Greedy search loop
    for _ in range(max_length):
        # Run the decoder model
        _decoder_output = _decoder_session.run(None, _input_data)

        # Update past_key_values with the current output
        if _decoder_output[1] is not None:
            _input_data['past_key_values.0.key'] = _decoder_output[1]
            _input_data['past_key_values.0.value'] = _decoder_output[2]
            _input_data['past_key_values.1.key'] = _decoder_output[3]
            _input_data['past_key_values.1.value'] = _decoder_output[4]
            _input_data['past_key_values.2.key'] = _decoder_output[5]
            _input_data['past_key_values.2.value'] = _decoder_output[6]
            _input_data['past_key_values.3.key'] = _decoder_output[7]
            _input_data['past_key_values.3.value'] = _decoder_output[8]
            _input_data['past_key_values.4.key'] = _decoder_output[9]
            _input_data['past_key_values.4.value'] = _decoder_output[10]
            _input_data['past_key_values.5.key'] = _decoder_output[11]
            _input_data['past_key_values.5.value'] = _decoder_output[12]
            _input_data['past_key_values.6.key'] = _decoder_output[13]
            _input_data['past_key_values.6.value'] = _decoder_output[14]
            _input_data['past_key_values.7.key'] = _decoder_output[15]
            _input_data['past_key_values.7.value'] = _decoder_output[16]
            _input_data['past_key_values.8.key'] = _decoder_output[17]
            _input_data['past_key_values.8.value'] = _decoder_output[18]
            _input_data['past_key_values.9.key'] = _decoder_output[19]
            _input_data['past_key_values.9.value'] = _decoder_output[20]
            _input_data['past_key_values.10.key'] = _decoder_output[21]
            _input_data['past_key_values.10.value'] = _decoder_output[22]
            _input_data['past_key_values.11.key'] = _decoder_output[23]
            _input_data['past_key_values.11.value'] = _decoder_output[24]
            _input_data['use_cache_branch'] = [True]

        # Extract the logits and apply softmax
        _logits = _decoder_output[0]
        # _probabilities = np.exp(_logits) / np.sum(np.exp(_logits), axis=-1, keepdims=True)

        # Get the token with the highest probability
        # next_token_id = np.argmax(_probabilities[:, -1, :], axis=-1).flatten()[0]
        next_token_id = np.argmax(_logits)

        # Append the token to the list
        generated_tokens.append(next_token_id)

        # Prepare the input for the next iteration
        _input_data['input_ids'] = np.array([[next_token_id]])

        # Check if EOS token is generated
        if next_token_id == _trg_tokenizer.eos_token_id:
            break

    # Decode the generated tokens into text
    print(f'generated_tokens: {generated_tokens}')
    _generated_text = _trg_tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return _generated_text


def translate(text):
    print(f'Translating text: {text}')

    encoded_input = src_tokenizer.encode_plus(text)

    input_ids = encoded_input["input_ids"]
    input_ids = np.expand_dims(input_ids, axis=0)
    input_ids = input_ids.astype(np.int64)
    print(f'input_ids: {input_ids}')

    attention_mask = encoded_input["attention_mask"]
    attention_mask = np.expand_dims(attention_mask, axis=0)
    attention_mask = attention_mask.astype(np.int64)

    batch_size, past_sequence_length = input_ids.shape[0], input_ids.shape[1]
    # np.zeros((batch_size, 12, past_sequence_length, 64), dtype=np.float32)

    input_data = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    encoder_output = encoder_session.run(None, input_data)

    output = encoder_output[0]

    output_data = {
        "input_ids": input_ids,
        "encoder_hidden_states": output,
        "use_cache_branch": [False],
        "past_key_values.0.key": None,  # np.zeros((batch_size, 12, past_sequence_length, 64), dtype=np.float32)
        "past_key_values.0.value": None,
        "past_key_values.1.key": None,
        "past_key_values.1.value": None,
        "past_key_values.2.key": None,
        "past_key_values.2.value": None,
        "past_key_values.3.key": None,
        "past_key_values.3.value": None,
        "past_key_values.4.key": None,
        "past_key_values.4.value": None,
        "past_key_values.5.key": None,
        "past_key_values.5.value": None,
        "past_key_values.6.key": None,
        "past_key_values.6.value": None,
        "past_key_values.7.key": None,
        "past_key_values.7.value": None,
        "past_key_values.8.key": None,
        "past_key_values.8.value": None,
        "past_key_values.9.key": None,
        "past_key_values.9.value": None,
        "past_key_values.10.key": None,
        "past_key_values.10.value": None,
        "past_key_values.11.key": None,
        "past_key_values.11.value": None,
    }

    # Use Greedy Search to get the most probable token ids
    generated_text = greedy_search(output_data, encoder_session, decoder_session, trg_tokenizer)

    return generated_text


texts = [
    "逃げろ!",  # Should be "도망쳐!"
    "初めまして.",  # "반가워요"
    "よろしくお願いします.",  # "잘 부탁드립니다."
    "ギルガメッシュ討伐戦",  # "길가메쉬 토벌전"
    "ギルガメッシュ討伐戦に行ってきます。一緒に行きましょうか？",  # "길가메쉬 토벌전에 갑니다. 같이 가실래요?"
    "夜になりました",  # "밤이 되었습니다"

    # why this text is not working properly?
    "ご飯を食べましょう."  # Should be "음, 이제 식사도 해볼까요"
    # But it is "음, 음, 음, 음, 음, 음, 음, 음, 음, 음, 음, 음, 음, 음, 음, 음, 음, 음, 음, 음, 음, 음, 음, 음, 음,"
]

for text in texts:
    print(translate(text))
    print()
