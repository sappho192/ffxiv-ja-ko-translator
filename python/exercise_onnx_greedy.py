from onnxruntime import InferenceSession, SessionOptions
from transformers import BertJapaneseTokenizer, PreTrainedTokenizerFast
import numpy as np
import os

encoder_model_name = "cl-tohoku/bert-base-japanese-v2"
decoder_model_name = "skt/kogpt2-base-v2"

src_tokenizer = BertJapaneseTokenizer.from_pretrained(encoder_model_name)
trg_tokenizer = PreTrainedTokenizerFast.from_pretrained(decoder_model_name)
trg_tokenizer.bos_token_id = 1
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

    num_pkv = 4  # Set the number of past key values
    past_key_values = None
    # Greedy search loop
    for _ in range(max_length):
        # Run the decoder model
        _decoder_output, past_key_values = onnx_forward(_decoder_session, _input_data, past_key_values)

        # Extract the logits and apply softmax
        _logits = _decoder_output[0]
        # _probabilities = np.exp(_logits) / np.sum(np.exp(_logits), axis=-1, keepdims=True)

        # Update past_key_values with the current output
        # if _input_data['use_cache_branch'][0] is False:
        #     # pack them
        #     out_past_key_values = tuple(
        #         _decoder_output[i: i + num_pkv] for i in range(0, len(_decoder_output), num_pkv)
        #     )
        # else:
        #     num_layers = len(_decoder_output) // 4  # Assuming 4 past_key_values per layer
        #     for i in range(num_layers):
        #         # Update self-attention key/value pairs
        #         _input_data[f'past_key_values.{2*i}.key'] = _decoder_output[4*i + 1]
        #         _input_data[f'past_key_values.{2*i}.value'] = _decoder_output[4*i + 2]
        #         # Cross-attention key/value pairs remain constant, no need to update
        # _input_data['use_cache_branch'] = [True]

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


def onnx_forward(_decoder_session, _input_data, _past_key_values, num_pkv=4):
    if _past_key_values is None:
        # initialize past_key_values
        # _input_data["past_key_values.0.key"] = empty_kv(1, 1)
        for key in _input_data:
            if key.startswith("past_key_values"):
                _input_data[key] = empty_kv(1, 1)
    else:  # Flatten the past_key_values
        _past_key_values = tuple(
            past_key_value for pkv_per_layer in _past_key_values for past_key_value in pkv_per_layer
        )
        i = 0
        for key in _input_data:
            if key.startswith("past_key_values"):
                if key.endswith(".key"):
                    _input_data[key] = _past_key_values[i]
                elif key.endswith(".value"):
                    _input_data[key] = _past_key_values[i]
                i += 1

        _input_data["use_cache_branch"] = [True]

    _decoder_output = _decoder_session.run(None, _input_data)
    # out_past_key_values = tuple(_input_data[key] for key in _input_data if key.startswith("past_key_values"))
    out_past_key_values = tuple(_decoder_output[1:])
    if _input_data["use_cache_branch"][0] is False:  # pack them to 4 units
        out_past_key_values = tuple(
            out_past_key_values[i: i + num_pkv] for i in range(0, len(out_past_key_values), num_pkv)
        )
    elif _input_data["use_cache_branch"][0] is True:
        out_past_key_values = tuple(
            out_past_key_values[i: i + 2] + _past_key_values[i + 2: i + 4]
            for i in range(0, len(out_past_key_values), num_pkv)
        )

    return _decoder_output, out_past_key_values


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
    past_sequence_length = 1  # Forcing assignment as 1
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
        "past_key_values.0.key": None,
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


def empty_kv(batch_size, past_sequence_length):
    return np.zeros((batch_size, 12, past_sequence_length, 64), dtype=np.float32)


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
