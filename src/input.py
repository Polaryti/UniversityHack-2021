import sys
import pandas

if __name__ == "__main__":
    dataset = pandas.read_csv(filepath_or_buffer = sys.argv[1], sep = '|')
    dataset.drop_duplicates(inplace=True)
    dataset.to_csv(index = False, path_or_buf= sys.argv[1].replace('.txt', '') + ".csv", sep='|')