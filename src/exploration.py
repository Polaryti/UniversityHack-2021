import sys
import pandas as pd
import numpy as np

if __name__ == "__main__":
    df = pd.read_csv(filepath_or_buffer = sys.argv[1], sep = '|')

    print(np.max(df['antiguedad']))
    print(np.min(df['antiguedad']))
    # for column in df:
    #     print(df[column].value_counts(dropna=False))
    #     print('\n')