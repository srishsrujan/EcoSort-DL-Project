# Failure Log

This file records the reasoning for the highest-confidence misclassifications in the held-out test set. The reasons below are based on the dominant visual cause visible in each error pattern and are intended to document failure modes without inventing new data.

| Image | True class | Predicted | Confidence | Reason |
|---|---|---|---:|---|
| `trash_142.jpg` | dry | recyclable | 0.993 | mixed waste |
| `paper_284.jpg` | dry | recyclable | 0.988 | ambiguous class |
| `plastic_58.jpg` | recyclable | dry | 0.982 | background |
| `biological_33.jpg` | wet | dry | 0.980 | ambiguous class |
| `trash_238.jpg` | dry | recyclable | 0.977 | mixed waste |
| `paper_627.jpg` | dry | recyclable | 0.976 | background |
| `cardboard_213.jpg` | dry | e-waste | 0.974 | object too small |
| `trash_334.jpg` | dry | e-waste | 0.968 | mixed waste |
| `battery_62.jpg` | e-waste | recyclable | 0.960 | reflection |
| `metal_850.jpg` | recyclable | dry | 0.949 | reflection |
| `shoes_1291.jpg` | dry | wet | 0.949 | ambiguous class |
| `paper_1186.jpg` | dry | recyclable | 0.944 | background |
| `trash_209.jpg` | dry | recyclable | 0.940 | mixed waste |
| `glass_1407.jpg` | recyclable | dry | 0.939 | reflection |
| `cardboard_559.jpg` | dry | recyclable | 0.930 | lighting |
| `plastic_517.jpg` | recyclable | dry | 0.930 | background |
| `plastic_1533.jpg` | recyclable | dry | 0.927 | lighting |
| `metal_796.jpg` | recyclable | dry | 0.917 | reflection |
| `biological_251.jpg` | wet | dry | 0.913 | ambiguous class |
| `paper_422.jpg` | dry | recyclable | 0.909 | background |
