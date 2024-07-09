import matplotlib.pyplot as plt
import pandas as pd


def main():
    df = pd.read_csv("state_data.csv")
    print(df.columns)
    df.plot(x="Time (ps)", y="Temperature (K)")
    plt.show()


if __name__ == '__main__':
    main()
