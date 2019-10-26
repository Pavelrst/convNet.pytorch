from args import comparisonal_histograms_args
import pandas as pd
import os
from os.path import join
import matplotlib.pyplot as plt

def separate(col_names):
    x_cols = []
    y_cols = []
    for idx in range(len(col_names)):
        if idx % 2 == 0:
            x_cols.append(col_names[idx])
        else:
            y_cols.append(col_names[idx])

    return x_cols, y_cols

def cut_outliers(x1, x2, y1, y2, cut = 100):
    x1 = x1[cut:]
    x1 = x1[:-cut]

    y1 = y1[cut:]
    y1 = y1[:-cut]

    x2 = x2[cut:]
    x2 = x2[:-cut]

    y2 = y2[cut:]
    y2 = y2[:-cut]

    return x1, x2, y1, y2

def generate(args):
    pd1 = pd.read_csv(join(args.path1, args.csvname))
    pd2 = pd.read_csv(join(args.path2, args.csvname))

    x_cols, y_cols = separate(list(pd1.columns.values)[1:])

    argfile1 = join(args.path1, args.argfile_name)
    argfile2 = join(args.path2, args.argfile_name)

    with open(argfile1, 'r') as f:
        content = f.readlines()
    for row in content:
        if 'eval_path' in row:
            evalpath1 = row

    with open(argfile2, 'r') as f:
        content = f.readlines()
    for row in content:
        if 'eval_path' in row:
            evalpath2 = row

    for x_col, y_col in zip(x_cols, y_cols):
        x_data1 = pd1[x_col]
        y_data1 = pd1[y_col]

        x_data2 = pd2[x_col]
        y_data2 = pd2[y_col]

        x_data1, x_data2, y_data1, y_data2 = cut_outliers(x_data1, x_data2, y_data1, y_data2)

        plt.plot(x_data1, y_data1, label=evalpath1)
        plt.plot(x_data2, y_data2, label=evalpath2)
        plt.legend()
        plt.show()

    return

def main():
    args = comparisonal_histograms_args()
    generate(args)
    return

main()