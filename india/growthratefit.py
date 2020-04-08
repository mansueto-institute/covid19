from pathlib import Path
from typing import Optional, Tuple
import warnings

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

def assume_missing_0(df: pd.DataFrame, col: str):
    return df[col] if col in df.columns else 0

def run_analysis(df: pd.DataFrame, 
        state: Optional[str] = None, 
        drop_kl3: bool  = True, # whether to drop the Feb cases in Kerala 
        window: int = 3, 
        infectious_period: float = 4.5,
        note: Optional[str] = None, 
        show_plots: bool = False) -> Tuple[int, int, int, int]:

    # filter data as needed and set up filename components
    if state and state.replace(" ", "").lower() not in ("all", "allstate", "allstates"):
        df = df[df["detected state"] == state]
        label = state.replace(" ", "_").lower()
    else: 
        state = "All States"
        label = "allstates"
    if drop_kl3:
        df = df[df["date announced"] > "2020/02/29"]
    
    if len(df) < 3:
        return 0, None, None, None
    
    note = '_' + note if note else ''
    
    output = Path(__file__).parent/"plots"

    # calculate daily totals and growth rate
    totals = df.groupby(["status change date", "current status"])["patient number"].count().unstack().fillna(0)
    totals["date"]     = totals.index
    totals["time"]     = (totals["date"] - totals["date"].min()).dt.days
    totals["logdelta"] = np.log(assume_missing_0(totals, "Hospitalized") - assume_missing_0(totals, "Recovered") -  assume_missing_0(totals, "Deceased"))
    
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
    growthrates["date"]         = growthrates.index
    growthrates["days"]         = totals.time

    # extrapolate growth rate into the future
    predrates = growthrates.iloc[-5:].copy()
    predrates["days"] -= predrates["days"].min()
    pred = sm.OLS.from_formula("gradient ~ days", data = predrates).fit()
    pred_intercept, pred_gradient = pred.params
    pred_se = pred.bse[1]
    days_to_critical  = int(-pred_intercept/pred_gradient)
    days_to_criticalM = int(-pred_intercept/(pred_gradient + 2 * pred_se))
    days_to_criticalm = int(-pred_intercept/(pred_gradient - 2 * pred_se))

    # figure: log delta vs time
    fig, ax = plt.subplots()
    totals.plot(y = "logdelta", ax = ax, label = "log(confirmed - recovered - dead)")
    plt.xlabel("Date")
    plt.ylabel("Daily Net Cases")
    plt.title(f"Cases over Time ({state})")
    plt.tight_layout()
    plt.savefig(output/f"cases_over_time_{label}{note}.png", dpi = 600)

    # figure: extrapolation
    # t0 = growthrates.iloc[-5:].days.max()
    # t = np.arange(t0, t0 + max(days_to_criticalm, days_to_criticalM) + 1)
    # pred_lower = pred_intercept + t * (pred_gradient - 2 * pred_se)
    # pred_upper = pred_intercept + t * (pred_gradient + 2 * pred_se)
    
    fig, ax = plt.subplots()
    # plt.fill_between(t, pred_upper, pred_lower, alpha = 0.3)
    
    plt.plot(growthrates.days, growthrates.gradient, 'ro-', alpha = 0.6)
    plt.fill_between(growthrates.days, growthrates.egrowthratem, growthrates.egrowthrateM, color='gray', alpha=0.3)
    plt.xlabel("Days of Outbreak")
    plt.ylabel("Growth Rate")
    plt.title(state)
    plt.tight_layout()
    plt.savefig(output/f"extrapolation_{label}{note}.png", dpi = 600)

    # figure: reproductive rate vs critical level 
    fig, ax = plt.subplots()
    plt.plot(growthrates.date, growthrates.R, 'ro-', alpha = 0.6)
    plt.fill_between(growthrates.date, growthrates.Rm, growthrates.RM, color='gray', alpha=0.3)
    plt.xlabel("Date")
    plt.ylabel("Reproductive Rate")
    plt.title(state)
    plt.tight_layout()
    plt.savefig(output/f"reproductive_rate_{label}{note}.png", dpi = 600)

    if show_plots: 
        plt.show()
    plt.close("all")

    return (len(totals.index), days_to_critical, days_to_criticalm, days_to_criticalM)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    seaborn.set_style('darkgrid')
    
    root = Path(__file__).parent
    df = load_data(root/"india_case_data_resave.csv", reduced = True)

    state_dct = []
    print("Running analysis for: ")
    for state in ["All States"] + list(df["detected state"].unique()):
        try:
            n, dtc, dtcm, dtcM = run_analysis(df, state)
            if dtc:
                print(f"  + {state} ({n} dates total, {dtc} days to critical)")
                state_dct.append([state, n, dtc, dtcm, dtcM])
            else: 
                print(f"  + {state} (insufficient data)")
        except Exception as e:
            print(f"  + {state} (insufficient data {e})")
    
    pd.DataFrame(state_dct, columns = ["state", "n", "dct", "dctm", "dctM"]).set_index("state").to_csv(root/"state_dct.csv")
        