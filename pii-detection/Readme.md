# Use Pre-trained Models

- Install SimpleTransformers:
```bash
pip install simpletransformers
```

- Place pre-trained models folder at: `pretrained`.
  - https://www.dropbox.com/scl/fi/kijxq419hxd0pevxfiszp/pretrained.zip?rlkey=qtofftsveyo8uiwqmypwh0ese&dl=0

- Predict PIIs using the pre-trained models:
```python
from simpletransformers.ner import NERModel

model_name = "bert"

model = NERModel(
        model_name, f"pretrained/{model_name}",
        use_cuda=False  # remove for GPU
    )

predictions, _ = model.predict([text])
```
