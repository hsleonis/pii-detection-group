# Use Pre-trained Models

- Install SimpleTransformers:
```bash
pip install simpletransformers
```

- Place pre-trained models folder at: `pretrained`.

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
