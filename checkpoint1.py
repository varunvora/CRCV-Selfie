import pandas as pd
import matplotlib.pyplot as plt

from common import col

def analyze_dataset(dataset_file_name="./Selfie-dataset/selfie_dataset.txt"):
    global col
    select_col = ['female', 'baby', 'child', 'teenager', 'youth', 'middleAge', 'senior']

    data_df = pd.read_csv(dataset_file_name, sep="\s+", names=list(col.keys()))
    select_df = data_df[select_col]
    print("FIRST 10", select_df[:10], sep="\n")
    print("\n")
    print("LAST 10", select_df[-10:], sep="\n")

    for col in select_col:
        temp = select_df[col].value_counts()
        count = [0, 0, 0]
        count[0], count[2] = temp[-1], temp[1]
        if 0 in temp:
            count[1] = temp[0]
        df = pd.DataFrame({"x": [-1, 0, 1], "count": count})
        # ax = df.plot.bar(x="x", y="count", rot=0, title=col, )
        yield df, col


if __name__ == "__main__":
    analyze_dataset()

