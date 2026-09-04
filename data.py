from datasets import load_dataset
ds = load_dataset("HTW-KI-Werkstatt/RamanBench", "wheat_lines", split="train")
df = ds.to_pandas()
