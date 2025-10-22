
import json
from io import StringIO

import pandas as pd


def extract_overview_table_from_notebook(path: str) -> pd.DataFrame:
    """Load cell with tag 'result' from a notebook and sort the resulting dataframe.

    Parameters
    ----------
    path
        path to the notebook
    """

    with open(path, 'r') as o:
        data = json.load(o)

    target_cell = next((cell for cell in data.get("cells", []) if cell.get(
        "metadata", {}).get("result", {}) == "overview"), None)
    html_list = target_cell['outputs'][0]['data']['text/html']
    html_string = ''.join(html_list)
    dfs = pd.read_html(StringIO(html_string), index_col=0)
    df = dfs[0]
    df['season_length'] = df['season_length'].apply(lambda x: pd.eval(x))
    return df.sort_values(by="name").reset_index(drop=True)
