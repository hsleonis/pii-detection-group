# Documntation
## Install Transformers:
```python
pip install simpletransformers
```

## Use the pre-trained Model:
```python
from transformers import pipeline

pii_detect = pipeline("ner", model="Leonis/xlnet-ner-hpii")

text = "My name is Sarah and I live in London. My email is sarah@enron.de and phone number is (490) 555‑0100"

pii_detect(text)
```

## Output:
`
[{'entity': 'U-PERSON',
  'score': 0.999969,
  'index': 3,
  'word': '▁Sarah',
  'start': 11,
  'end': 16},
 {'entity': 'U-LOC',
  'score': 0.9999057,
  'index': 8,
  'word': '▁London',
  'start': 31,
  'end': 37},
 {'entity': 'U-EMAIL',
  'score': 0.999902,
  'index': 13,
  'word': '▁',
  'start': 51,
  'end': 52},
 {'entity': 'U-EMAIL',
  'score': 0.9999032,
  'index': 14,
  'word': 'sa',
  'start': 51,
  'end': 53},
 {'entity': 'U-EMAIL',
  'score': 0.9997837,
  'index': 15,
  'word': '.',
  'start': 53,
  'end': 54},
 {'entity': 'U-EMAIL',
  'score': 0.9994962,
  'index': 16,
  'word': 'rah',
  'start': 54,
  'end': 57},
 {'entity': 'U-EMAIL',
  'score': 0.99498594,
  'index': 17,
  'word': '@',
  'start': 57,
  'end': 58},
 {'entity': 'U-EMAIL',
  'score': 0.6740328,
  'index': 18,
  'word': 'en',
  'start': 58,
  'end': 60}]
`

