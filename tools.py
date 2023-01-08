import numpy as np


def create_dict(df=None, keys_list=None):
    # create a dictionary of the key-values provided in the list
    row_idx, col_idx = df.where(df.isin(keys_list)).stack().index.codes
    list_row_idx = list(np.unique(row_idx))
    # add the value column, heuristic is that the value column is preceded by the key and that list_col_idx is the key
    list_col_idx = [np.argmax(np.bincount(col_idx))]
    list_row_idx = [i for i, condition in zip(list_row_idx, list_col_idx == col_idx) if condition]
    list_col_idx = list_col_idx + [list_col_idx[0] + 1]
    filter_series = df.loc[list_row_idx, list_col_idx].set_index(list_col_idx[0]).squeeze()
    out_dict = filter_series.to_dict()
    return out_dict


def create_df(df=None, columns_list=None, min_row=None, max_row=None):
    # create a DataFrame from a list of columns in a sheet (that might be a sub table) and end the dataframe on the
    # last row of values
    row_idx, col_idx = df.where(df.isin(columns_list)).stack().index.codes
    if max_row is None:
        max_row = df.index.max() + 1
    if min_row is None:
        min_row = df.index.min() + 1
    filter_list = [i for i, x in enumerate(row_idx) if min_row <= x <= max_row]
    list_col_idx = list(np.unique(col_idx[filter_list]))
    first_row_idx = list(np.unique(row_idx[filter_list]))[0]
    filter_col_df = df.loc[list(range(first_row_idx, max_row)), list_col_idx]
    # set first_null_row to max_row if there are no null values at the end of the dataframe
    try:
        first_null_row = filter_col_df[filter_col_df.isnull().all(axis=1)].index[0] - first_row_idx
    except:
        first_null_row = max_row
    out_df = filter_col_df[:first_null_row]
    out_df.columns = out_df.iloc[0]
    out_df = out_df.drop(out_df.index[0])
    return out_df


def divide_by(numerator, denominator):
    if denominator == 0:
        return 0
    else:
        return numerator / denominator