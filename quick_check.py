# quick_check_labels.py
import pickle, numpy as np
from utils.config import CONFIG

X,y = pickle.load(open(CONFIG["vector_cache"],"rb"))
print("ALL:", dict(zip(*np.unique(y, return_counts=True))))
