import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Récupère la data
liste = []
with open(r"/Users/lucashennecon/Documents/Cours_CS/Infonum/EI/X_train_update.csv", encoding="Macintosh") as file_name:
    for el in file_name:
        el = el[:-1]
        ele = el.split(',')
        liste.append(ele)

# Arrange la data en concaténant titre et description
liste = liste[1:]
n = len(liste)
dataset = []
for i in range(n):
    if len(liste[i]) == 5:
        dataset.append(liste[i][1]+liste[i][2])


# Crée le tf-idf:
tfIdfVectorizer = TfidfVectorizer(use_idf=True)
tfIdf = tfIdfVectorizer.fit_transform(dataset)
df = pd.DataFrame(tfIdf[0].T.todense(
), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
df = df.sort_values('TF-IDF', ascending=False)
print(df)
