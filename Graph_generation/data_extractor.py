###            words = line.split(.)
import pandas
import numpy as np
df = (pandas.read_csv('data_banknote_authentication.csv').to_numpy())
np.random.shuffle(df)

print(df)
