from spliteDataset import load_dataset
import json

tr, va, te = load_dataset()
print(json.dumps(tr[0], indent=2, default=str))
print(sorted({k for r in tr[:2000] for k in r}))
print(sorted({k for r in tr[:2000] for k in r['sender']}))