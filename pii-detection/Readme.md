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

# Dataset

The `dataset.zip` contains the complete annotated HPII dataset used in the project.
It has two (2) CSV files:
- Training contains 70% of EnronPII + 70% of WikiPII: *hpii_train.csv*
- Testing 30% of EnronPII + 30% of WikiPII: *hpii_test.csv*

# Entities

The pretrained models detect following PIIs:
- Name
- Location
- Email Address
- Phhone number

Each of these entities are predicted into specific tokens:
- `B` : Beginning of an entity
- `I` : Inside token of an entity
- `L` : End of an entity
- `U` : Single token entity
- `O` : Non-PII

