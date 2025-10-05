import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset("titanic")

sns.barplot(titanic.groupby("sex").count().survived, label="Total")
sns.barplot(titanic.groupby("sex").sum("survived").survived, label="Alive")
plt.legend()
plt.show()

sns.barplot(titanic.groupby("class").count().survived, label="Total")
sns.barplot(titanic.groupby("class").sum("survived").survived, label="Alive")
plt.legend()
plt.show()
