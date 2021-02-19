import sys
import pandas

if __name__ == "__main__":
    df = pandas.read_csv(filepath_or_buffer = sys.argv[1], sep = '|')
    df.drop_duplicates(inplace=True)
    df['fecha'] = df['fecha'].apply(lambda x: x.replace(' 0:00:00', ''))
    df.to_csv(index = False, path_or_buf= sys.argv[1].replace('.txt', '') + ".csv", sep='|')