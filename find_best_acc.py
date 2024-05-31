import argparse
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def draw_acc(acc1_values):
    # plt.figure(figsize=(6, 6))
    acc1_values = 100 - np.array(acc1_values)
    # print(acc1_values)
    num_epochs = len(acc1_values)
    lw = 1
    plt.plot(range(1, num_epochs + 1), acc1_values, "--", lw=lw, label="current")
    
    # plt.title("Acc of each epoch")
    plt.xlabel("Training Epoch")
    plt.ylabel("Top1 Error (%)")
    plt.legend(loc="center right")
    plt.grid(True)
    plt.savefig("model_acc.png")
    plt.clf()
    plt.close()
    

def find_max_acc1_from_log(file_path):
    try:
        with open(file_path, 'r') as file:
            log_data = file.read()

        # Extract all acc1 values using regex
        acc1_values = re.findall(r"Test:  Acc@1 (\d+\.\d+)", log_data)
        # print(acc1_values)
        acc1_values = list(map(float, acc1_values))
        print(acc1_values)
        draw_acc(acc1_values)

        # Convert to float and find the max value
        max_acc1 = max(acc1_values)
        max_index = acc1_values.index(max_acc1)
        return max_acc1, max_index
    except FileNotFoundError:
        return "File not found"
    except Exception as e:
        return f"An error occurred: {e}"

def main():
    parser = argparse.ArgumentParser(description="Find the maximum acc1 value from a log file.")
    parser.add_argument("file_path", type=str, help="Path to the log file")

    args = parser.parse_args()

    max_acc1_value, max_index = find_max_acc1_from_log(args.file_path)
    print(f"The maximum acc1 value in the log is: {max_acc1_value}, epoch: {max_index}")

if __name__ == "__main__":
    main()
