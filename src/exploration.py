import sys
import pandas as pd
import numpy as np

if __name__ == "__main__":
    df = pd.read_csv(filepath_or_buffer = sys.argv[1], sep = '|')
    print(df['dia_atipico'].value_counts())