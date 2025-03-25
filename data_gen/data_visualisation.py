import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_data(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    bucket_size: int,
):
    """
    Plot the data in a histogram with buckets of values

    Parameters:
    data (pd.DataFrame): The data to plot
    x_col (str): The column to plot on the x-axis
    y_col (str): The column to plot on the y-axis
    title (str): The title of the plot
    x_label (str): The label of the x-axis
    y_label (str): The label of the y-axis
    bucket_size (int): The size of each bucket

    """
    # Calculate the number of buckets
    num_buckets = int((data[x_col].max() - data[x_col].min()) / bucket_size) + 1

    # Create the histogram
    plt.hist(data[x_col], bins=num_buckets)

    # Set the title and labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Set the x-axis tick locations
    plt.xticks(
        range(int(data[x_col].min()), int(data[x_col].max()) + bucket_size, bucket_size)
    )

    # Show the plot
    plt.show()


def load_dataframe(file_name: str) -> pd.DataFrame:
    """
    Load a csv file into a pandas DataFrame

    Parameters
    ----------
    file_name : str
        The name of the file to load

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame
    """

    file_path = os.path.join(
        os.getcwd(), "data_gen", "incidence_matrix_dataframes", file_name
    )
    return pd.read_csv(file_path)


def main() -> None:

    data = load_dataframe("torus_dataset.csv")
    plot_data(
        data,
        "width_δ2",
        "y",
        "Distribution over faces - Torus",
        "Number of faces",
        "Frequency",
        bucket_size=50,
    )


if __name__ == "__main__":
    main()
