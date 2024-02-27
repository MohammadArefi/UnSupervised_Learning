import pandas as pd
df = pd.DataFrame({'A':[0,0,0,1,0], 'B':[0,0,1,0,0], 'C':[1,1,0,0,1], 'D':[0,1,0,1,1]})
import numpy as np
from pomegranate.bayesian_network import *  
model = BayesianNetwork.from_samples(df.to_numpy(), state_names=df.columns.values, algorithm='exact')
model.plot()