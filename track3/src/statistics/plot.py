import pickle
import matplotlib
matplotlib.use('Agg') # 
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import pickle
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Plot Log File to Curve")
    parser.add_argument("log_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    # 
    loss_list = readFromPickle(args.log_path)
    # 
    plot_and_save(loss_list, args.output_path)

def readFromPickle(log_path):
    with log_path.open("rb") as f:
        loss_list = pickle.load(f) 
    return loss_list

def plot_and_save(loss_list, output_path):
    losses = np.array(loss_list)
    plt.title("Loss to Epoch Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Cross Entropy Loss")
    plt.plot(range(len(losses)), losses)
    plt.savefig(str(output_path))
    plt.clf()


if __name__ == "__main__":
    main()