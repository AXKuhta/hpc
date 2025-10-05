import numpy as np
import pandas as pd
import seaborn as sns

planets = sns.load_dataset("planets")

result = planets.groupby([planets.method, planets.year//10]).sum().number

print(f"{'method':<30} decade number")

for (method, decade), number in zip(result.index, result.values):
	print(f"{method:<30} {decade*10} {number}")
