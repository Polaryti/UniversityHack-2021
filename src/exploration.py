import sys
import pandas
import numpy as np

if __name__ == "__main__":
    df = pandas.read_csv(filepath_or_buffer = sys.argv[1], sep = '|')
    df.info()