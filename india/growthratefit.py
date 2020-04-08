from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

import matplotlib.pyplot as plt
import seaborn

columns   = [
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
    "type of transmission"
}

# assuming analysis for data structure from COVID19-India saved as resaved, properly-quoted file
def load_data(datapath: Path, reduced: bool = False) -> pd.DataFrame: 
    return pd.read_csv(datapath, 
        skiprows    = 1, # supply fixed header in order to deal with Google Sheets export issues 
        names       = columns, 
        usecols     = (lambda _: _ not in drop_cols) if reduced else None,
        dayfirst    = True, # source data does not have consistent date format so cannot rely on inference
        parse_dates = ["date announced", "status change date"])


def run_analysis(df: pf.DataFrame, 
        state: Optional[str] = None, 
        drop_kl3: bool  = True, # whether to drop the Feb cases in Kerala 
        window: int = 3, 
        infectious_period: float = 4.5,
        note: Optional[str] = None, 
        show_plots: bool = False) -> Tuple[int, int, int]:

    # filter data as needed and set up filename components
    if state:
        df = df[df["detected state"] == state]
        label = state.replace(" ", "_").lower()
    else: 
        state = "All States"
        label = "allstates"
    if drop_kl3:
        df = df[df["date announced"] > "2020/02/29"]
    note = '_' + note if note else ''

    output = Path(__file__).parent/"plots"

    # calculate daily totals and growth rate
    totals = df.groupby(["status change date", "current status"])["patient number"].count().unstack().fillna(0)
    totals["date"]     = totals.index
    totals["time"]     = (totals["date"] - totals["date"].min()).dt.days
    totals["logdelta"] = np.log(totals["Hospitalized"] - totals["Recovered"] - totals["Deceased"])
    
    # run rolling regressions and get parameters
    model   = RollingOLS.from_formula(formula = "logdelta ~ time", window = window, data = totals)
    rolling = model.fit(method = "lstsq")
    
    growthrates = rolling.params.join(rolling.bse, rsuffix="_stderr")
    growthrates["rsq"] = rolling.rsquared
    growthrates.rename(lambda s: s.replace("time", "gradient").replace("const", "intercept"), axis = 1, inplace = True)

    # calculate growth rates
    growthrates["egrowthrateM"] = growthrates.gradient + 2 * growthrates.gradient_stderr
    growthrates["egrowthratem"] = growthrates.gradient - 2 * growthrates.gradient_stderr
    growthrates["R"]            = growthrates.gradient * infectious_period + 1
    growthrates["RM"]           = growthrates.gradient + 2 * growthrates.gradient_stderr * infectious_period + 1
    growthrates["Rm"]           = growthrates.gradient - 2 * growthrates.gradient_stderr * infectious_period + 1
    growthrates["days"]         = totals.time

    # TODO (satej): infectious vs time (show models and best fit from original code)

    # extrapolate growth rate into the future
    pred = sm.OLS.from_formula("gradient ~ days", data = growthrates.iloc[-5:]).fit()
    pred_intercept, pred_gradient = pred.params
    days_to_critical, days_to_criticalm, days_to_criticalM = map(int, -pred_intercept/(pred_gradient + pred.bse[1] * np.array([0, -2, 2])))

    # figure: log delta vs time
    fig, ax = plt.subplots()
    totals.plot(y = "logdelta", ax = ax, label = "log(confirmed - recovered - dead)")
    plt.xlabel("Date")
    plt.ylabel("Net Cases")
    plt.title(f"Net Cases over Time ({state})")
    plt.tight_layout()
    plt.savefig(output/f"cases_over_time_{label}{note}.png", dpi = 600)

    # figure: 

    if show_plots: plt.show()

    return (days_to_critical, days_to_criticalm, days_to_criticalM)

if __name__ == "__main__":
    seaborn.set_style('darkgrid')
    
    df = load_data(Path(__file__).parent/"india_case_data_resave.csv", reduced = True)

    print("Running analysis for: ")
    for state in [None] + list(df["detected state"].unique()):
        dtc, _, _ = run_analysis(df, state)
        print(f"  + {state if state else 'All States'} (days to critical: {dtc})")
        