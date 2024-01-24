from pretty_downloader import download

# If onnx2 directory doesn't exist, create it
import os
if not os.path.exists('onnx2'):
    os.makedirs('onnx2')

ONNX2_PATH_ROOT = 'onnx2'

download('https://huggingface.co/sappho192/ffxiv-ja-ko-translator/resolve/main/onnx/generation_config.json',
         file_path=ONNX2_PATH_ROOT)

