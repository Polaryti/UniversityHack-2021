import sys
import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    df = pd.read_csv(filepath_or_buffer = sys.argv[1], sep = '|')

    scaler = StandardScaler()