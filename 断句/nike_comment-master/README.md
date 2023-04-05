# tokenize 分句

## 训练

1. 运行`pip install -r requirements.txt`
2. 把数据放到`data/commentsList.txt`，每行一段，一段中用`|`分句
3. 运行`python preprocess.py`会在`data`下生成`processed.pkl`
4. 运行`python train.py`会在`checkpoint`下生成模型文件

## 推断

```python
from evaluation import Splitter
import torch as th
model = th.load("checkpoint/conv1d-100k-001.pkl")
splitter = Splitter("bert-base-chinese", model)
splitter("缓震很好颜值很高我很喜欢")
```