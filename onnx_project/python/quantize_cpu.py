# Note that this script works until onnxruntime 1.16.3.
# 1.17.0 will not work because of the following error:
# https://github.com/huggingface/optimum/issues/1687

from optimum.onnxruntime import ORTQuantizer, ORTModelForSeq2SeqLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig

PATH_TO_QUANTIZED = "D:\\MODEL\\ffxiv-ja-ko-translator\\quantized\\cpu\\avx2\\dynamic"
# create directory if not exists
import os
os.makedirs(PATH_TO_QUANTIZED, exist_ok=True)

model_id = "sappho192/ffxiv-ja-ko-translator"
onnx_model = ORTModelForSeq2SeqLM.from_pretrained(model_id, subfolder="onnx")
# model_id = "D:\\CODE\\ffxiv-ja-ko-translator\\onnx"
# onnx_model = ORTModelForSeq2SeqLM.from_pretrained(model_id)

model_dir = onnx_model.model_save_dir

encoder_quantizer = ORTQuantizer.from_pretrained(model_dir, file_name="encoder_model.onnx")
decoder_quantizer = ORTQuantizer.from_pretrained(model_dir, file_name="decoder_model.onnx")
decoder_wp_quantizer = ORTQuantizer.from_pretrained(model_dir, file_name="decoder_with_past_model.onnx")
# decoder_merged_quantizer = ORTQuantizer.from_pretrained(model_dir, file_name="decoder_model_merged.onnx")

quantizer = [encoder_quantizer, decoder_quantizer, decoder_wp_quantizer]
# quantizer = [encoder_quantizer, decoder_quantizer, decoder_wp_quantizer, decoder_merged_quantizer]

dqconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)

for q in quantizer:
    q.quantize(save_dir=PATH_TO_QUANTIZED, quantization_config=dqconfig)  # doctest: +IGNORE_RESULT
