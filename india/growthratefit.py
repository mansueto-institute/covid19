from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

import matplotlib.pyplot as plt
import seaborn

seaborn.set_style('darkgrid')

names = [
    "patient number",
    "state patient number",
    "date announced",
    "age bracket",
    "gender",
    "detected city",
    "detected district",
    "detected state",
    "current status",
    "notes",
    "contracted from which patient (suspected)",
    "nationality",
    "type of transmission",
    "status change date",
    "source_1",
    "source_2",
    "source_3",
    "backup note"
]

drop_cols = {
    "age bracket",
    "gender",
    "detected city",
    "detected district",
    "notes",
    "contracted from which patient (suspected)",
    "nationality",
    "source_1",
    "source_2",
    "source_3",
    "backup note",
    "type of transmission",
}

# assuming analysis for data structure from COVID19-India saved as resaved, properly-quoted file
def load_data(datapath: Path, reduced: bool = False) -> pd.DataFrame: 
    return pd.read_csv(datapath, 
        skiprows    = 1, 
        names       = names, 
        usecols     = (lambda _: _ not in drop_cols) if reduced else None,
        dayfirst    = True,
        parse_dates = ["date announced", "status change date"])


def run_analysis(df: pf.DataFrame, 
        state: Optional[str] = None, 
        drop_kl3: bool  = True, # whether to drop the Feb cases in Kerala 
        window: int = 3, 
        infectious_period: float = 4.5,
        note: Optional[str] = None):

    # filter data as needed
    if state:
        df = df[df["detected state"] == state]
        label = state 
    else: 
        label = "all"
    if drop_kl3:
        df = df[df["date announced"] > "2020/02/29"]

    # calculate daily totals and growth rate
    totals = df.groupby(["status change date", "current status"])["patient number"].count().unstack().fillna(0)
    totals["date"]     = totals.index
    totals["time"]     = (totals["date"] - totals["date"].min()).dt.days
    totals["logdelta"] = np.log(totals["Hospitalized"] - totals["Recovered"] - totals["Deceased"])
    
    # run rolling regressions and get parameters
    rolling = RollingOLS(
        endog  = totals["logdelta"], 
        exog   = sm.add_constant(totals["time"]), 
        window = window
    ).fit(method = "lstsq")
    
    params = rolling.params.join(rolling.bse, rsuffix="_stderr")
    params["rsq"] = rolling.rsquared
    params.rename(columns = {
        "time"        : "gradient",
        "const"       : "intercept",
        "time_stderr" : "gradient_stderr",
        "const_stderr": "intercept_stderr"
    }, inplace = True)

    # calculate growth rates
    params["egrowthrateM"] = params.gradient + 2 * params.gradient_stderr
    params["egrowthratem"] = params.gradient - 2 * params.gradient_stderr
    params["R"]            = params.gradient * infectious_period + 1
    params["RM"]           = params.gradient + 2 * params.gradient_stderr * infectious_period + 1
    params["Rm"]           = params.gradient - 2 * params.gradient_stderr * infectious_period + 1

    # to do: infectious vs time (show models and best fit from original code)

