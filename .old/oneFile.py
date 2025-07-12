import argparse 
import os 
import sys
import pandas as pd 
import numpy as np 

def read_hv(df, N, t): # t as an int or str? right now t is a float 
    row = df.loc[str(t), : N ** 2 + 2]
    V = np.array(row[-3:])
    h = np.array(row[ : -3]).reshape(N, N)
    return h, V

def main(args):
    folder = args.folder
    sfL = next(os.walk(folder))[1]
    d = {}
    for sf in sfL:
        t = float(sf)
        h = np.loadtxt(os.path.join(folder, sf, "h.txt")).flatten()
        V = np.loadtxt(os.path.join(folder, sf, "V.txt")).flatten()
        row = np.append(h, V)
        d[sf] = row
    df = pd.DataFrame(d).T
    df["t"] = df.index.astype(float)
    df.to_csv(folder+".csv", index = True)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Compacting one file', description='Converting \'h.txt\' into a csv file')
    parser.add_argument("folder", help = "folder to be converted into a csv file")

    args = parser.parse_args()
    main(args)