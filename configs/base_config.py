import datetime
import os
import torch

"""
defining classes for grouping parameters.

classes:
--------
    settings
    paths
    env_params
"""

class settings:
    """
    Defining general settings for the whole run and global variables.
    """
    # ---------------SET MANUALLY---------------
    # dataset used:
    DATASET = "US_stocks_WDB_full" #"done_data"
    #DATASET = "JP_stocks_WDB" # todo: rm

    ### strategy mode to be run
    STRATEGY_MODE = "ppoCustomBase"

    REWARD_MEASURE = "addPFVal" # additional portfolio value, = change in portfolio value as a reward
    #REWARD_MEASURE = "logU" # log utility of new / old value, in oder to "smooth out" larger rewards
    #REWARD_MEASURE = "SR7" # sharpe ratio, over 7 days # subtracting a volatility measure # todo: rm
    #REWARD_MEASURE = "semvarPenalty"

    RETRAIN_DATA = False # = saving trained agent after each run and continue training only on the next train data chunk, using pre-trained agent (faster)
    #RETRAIN_DATA = True # = when training again on the whole training dataset for each episode

    ### Set dates
    # train
    STARTDATE_TRAIN = 20090101 #20141001 #20090102  # Note: this is also the "global startdate"
    ENDDATE_TRAIN = 20151001
    # validation (only needed for get_data_params in preprocessing)
    #STARTDATE_VALIDATION = 20160101 #20151001
    #ENDDATE_VALIDATION = #20200707
    # trading starts on:     # 2016/01/01 is the date that real trading starts
    #STARTDATE_TRADE = 20160104
    #ENDDATE_TRADE = None
    # backtesting
    STARTDATE_BACKTESTING_BULL = 20070605
    ENDDATE_BACKTESTING_BULL = 20070904 # there is no 2./3. sept
    STARTDATE_BACKTESTING_BEAR = 20070904
    ENDDATE_BACKTESTING_BEAR = 20071203

    ### set rollover window; since we are doing rolling window / extended window cross validation for time series
    # 63 days = 3 months of each 21 trading days (common exchanges don't trade on weekends, need to change for crypto)
    ROLL_WINDOW = 63
    VALIDATION_WINDOW = 63
    TESTING_WINDOW = 63

    # ---------------LEAVE---------------
    ### define 10 randomly picked numbers to be used for seeding
    SEEDS_LIST = [11112] #0, 5, 23, 7774, 11112]#,  45252, 80923, 223445, 444110]
    SEED = None # placeholder, will be overwritten in run file)

    ### returns current timestamp, mainly used for naming directories/ printout / logging to .txt
    NOW = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

    # this is going to be in the run folder name
    if RETRAIN_DATA:
        # if we retrain data, run models "long" (simply because it takes longer)
        RUN_MODE = "lng" # for "long"
    else:
        # if we do not retrain data, run mode is short (the run takes less long)
        RUN_MODE = "st" # for "short"

class paths:
    # ---------------LEAVE---------------
    # data paths
    DATA_PATH = "data"
    RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")
    PREPROCESSED_DATA_PATH = os.path.join(DATA_PATH, "preprocessed")

    # trained models and results path
    TRAINED_MODELS_PATH = "trained_models"
    RESULTS_PATH = "results"
    # names of sub-directories within results folder (based on the memories we save in the env.)
    SUBSUBDIR_NAMES = {"datadates": "datadates",
                       "cash_value": "cash_value",
                       "portfolio_value": "portfolio_value",
                       "rewards": "rewards",
                       "policy_actions": "policy_actions",
                       "policy_actions_trans": "policy_actions_trans",
                       "exercised_actions": "exercised_actions",
                       "asset_equity_weights": "asset_equity_weights",
                       "all_weights_cashAtEnd": "all_weights_cashAtEnd",
                       "transaction_cost": "transaction_cost",
                       "number_asset_holdings": "number_asset_holdings",
                       "sell_trades": "sell_trades",
                       "buy_trades": "buy_trades",
                       "state_memory": "state_memory",
                       "last_state": "last_state",
                       "backtest_bull": "backtest_bull",
                       "backtest_bear": "backtest_bear",
                       "training_performance": "training_performance",
                       }

    # ---------------LEAVE---------------
    PREPROCESSED_DATA_FILE = os.path.join(PREPROCESSED_DATA_PATH, f"{settings.DATASET}.csv")

