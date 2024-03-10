# Convert LORE to ONNX
该仓库用来转换[无线表格识别模型LORE](https://www.modelscope.cn/models/iic/cv_resnet-transformer_table-structure-recognition_lore/summary)为ONNX格式。

#### 1. Clone the source code.
```bash
git clone https://github.com/SWHL/ConvertLaTeXOCRToONNX.git
```

#### 2. Install env.
```bash
conda install --yes --file requirements.txt
```
#### 3. Run the demo, and the converted model is located in the `moodels` directory.
```bash
python main.py
```

#### 4. Install `lineless_table_rec`
```bash
pip install lineless_table_rec
```

#### 5. Use
```python
from pathlib import Path

from lineless_table_rec import LinelessTableRecognition

detect_path = "models/lore_detect.onnx"
process_path = "models/lore_process.onnx"
engine = LinelessTableRecognition(
    detect_model_path=detect_path, process_model_path=process_path
)

img_path = "images/lineless_table_recognition.jpg"
table_str, elapse = engine(img_path)

print(table_str)
print(elapse)

with open(f"{Path(img_path).stem}.html", "w", encoding="utf-8") as f:
    f.write(table_str)

print("ok")
```