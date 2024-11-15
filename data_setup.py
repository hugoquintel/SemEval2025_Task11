import pandas as pd
def read_data(current_path, data_type, track, language):
    data_path = current_path / 'data' / data_type / track / f'{language}.csv'
    return pd.read_csv(data_path)