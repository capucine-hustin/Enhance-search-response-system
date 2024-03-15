import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

liste=[]
with open(r"C:\Users\louis\OneDrive\Documents\CS\EI-e-commerce\X_train_update.csv", encoding="Macintosh") as file_name:
    for el in file_name :
        el=el[:-1]
        ele=el.split(',')
        liste.append(ele)

liste=liste[1:]
n=len(liste)
dataset=[]
for i in range(n):
    if len(liste[i])==5:
        dataset.append(liste[i][1]+liste[i][2])

