import os
from dataclasses import dataclass

@dataclass(frozen=True) # class objects immutable after instantiation
class DataPrepConfig:
    """Configs for data preprocessing.
    """

    data_source : str = "WDB"  # Wharton Data Base # TODO: was named database previously, refac and enable multiple db choices

    data_path: str = os.path.join("data")
    
    raw_data_path: os.path = os.path.join(data_path, "raw")
    intermediate_data_path: os.path = os.path.join(data_path, "intermediate")
    processed_data_path: os.path = os.path.join(data_path, "preprocessed")

    # stock data file paths, un-preprocessed
    raw_data_file_us: os.path = os.path.join(raw_data_path, "US_stocks_WDB.csv")
    raw_data_file_jp: os.path = os.path.join(raw_data_path, "JP_stocks_WDB.csv")

    # dummy data file (only for testing the algorithm, not real data)
    dummydata: os.path = os.path.join(intermediate_data_path, "dummydata.csv")
    
    def __post_init__(self):
        if self.data_source not in ['WDB']:
            raise ValueError(f"Unsupported data_source: {self.data_source}")
        
        paths = {
            "raw_data_path": self.raw_data_path, 
            "intermediate_data_path": self.intermediate_data_path, 
            "processed_data_path": self.processed_data_path
        }
        for path_name, path  in paths.items:
            if not os.path.exists(path):
                raise ValueError(f"Unsupported {path_name}: {path}")
                
        filepaths = {
            "raw_data_file_us": self.raw_data_file_us, 
            "raw_data_file_jp": self.raw_data_file_jp, 
            "dummydata": self.dummydata
        }
        for path_name, path in filepaths.items:
            if not os.path.isfile(path):
                raise ValueError(f"Unsupported {path_name}: {path}")             



class data_settings: # TODO: convert to dataclass
    """
    Define variables and settings for data preprocessing.
    """
    # ---------------SET MANUALLY---------------
    # DATA SOURCE AND DATA SET CODE
    DATABASE = "WDB"  # stands for Wharton Data Base
    COUNTRY = "US"

    ### CHOOSE WHICH ARE THE (MANDATORY) BASE COLUMNS; for Wharton DB: datadate, tic
    if DATABASE == "WDB":
        # adjcp (adjusted closing price) is a default column we need in the state space because we need it to calculate
        # the number of stocks we can buy with our limited budget
        MAIN_PRICE_COLUMN = "adjcp"
        # this is the column where we store the tickers, we do not need them in our state space but in order to
        # reformat the data set from long to wide format
        ASSET_NAME_COLUMN = "tic"
        # this is the date column, we need it to split the data set in different train / validation / test sets
        DATE_COLUMN = "datadate"
        # these are the columns which are not used for state representation
        BASE_DF_COLS = [DATE_COLUMN, ASSET_NAME_COLUMN]

    ### FOR DATA PREPROCESSING:
    # 1) Choose subset of columns to be loaded in from raw dataframe
    # depends on the dataset used (by default: Wharton Database)
    # needs to be tailored depending on dataset / data source
    RAW_DF_COLS_SUBSET = BASE_DF_COLS + ['prccd', 'ajexdi', 'prcod', 'prchd', 'prcld', 'cshtrd']
    # 2) Choose which new columns should be created as intermediate step based on RAW_DF_COLS_SUBSET
    # by default: using Wharton DB data
    NEW_COLS_SUBSET = ['adjcp', 'open', 'high', 'low', 'volume']

    ### PROVIDE NAMES OF ALL FEATURES / INDICATORS GIVEN DATASET COLUMN NAMES
    PRICE_FEATURES = [MAIN_PRICE_COLUMN]
    TECH_INDICATORS = ["macd", "rsi_21", "cci_21", "dx_21"]#, "obv"] # technical indicators for momentum, obv instead of raw "volume"
    RETURNS_FEATURES = ["log_return_daily"] # log returns because they are a bit less "extreme" when they are larger and since we have daily returns this could be practical
    RISK_INDICATORS = ["ret_vola_21d"] # 21 days volatility and daily vix (divide by 100)
    SINGLE_FEATURES = ["vixDiv100"] # not attached to a certain asset

    # only applied if lstm net arch chosen
    LSTM_FEATURES = RETURNS_FEATURES + RISK_INDICATORS + SINGLE_FEATURES

    # CHOOSE FEATURES MODE, BASED ON WHICH THE FEATURES LIST IS CREATED (SEE BELOW)
    FEATURES_MODE = "fm7"

    # ---------------LEAVE---------------
    if FEATURES_MODE == "fm1":
        FEATURES_LIST = PRICE_FEATURES + RETURNS_FEATURES
        SINGLE_FEATURES_LIST = []
        LSTM_FEATURES_LIST = LSTM_FEATURES
    elif FEATURES_MODE == "fm2": # features version of the ensemble paper
        FEATURES_LIST = PRICE_FEATURES + TECH_INDICATORS #+ RETURNS_FEATURES
        SINGLE_FEATURES_LIST = []
        LSTM_FEATURES_LIST = LSTM_FEATURES
    elif FEATURES_MODE == "fm3":
        FEATURES_LIST = PRICE_FEATURES + RETURNS_FEATURES + TECH_INDICATORS + RISK_INDICATORS
        SINGLE_FEATURES_LIST = SINGLE_FEATURES
        LSTM_FEATURES_LIST = LSTM_FEATURES
    elif FEATURES_MODE == "fm4":
        FEATURES_LIST = PRICE_FEATURES + TECH_INDICATORS + RISK_INDICATORS#+ RETURNS_FEATURES
        SINGLE_FEATURES_LIST = SINGLE_FEATURES
        LSTM_FEATURES_LIST = LSTM_FEATURES
    elif FEATURES_MODE == "fm5":
        FEATURES_LIST = PRICE_FEATURES + TECH_INDICATORS + RETURNS_FEATURES#+ RETURNS_FEATURES
        SINGLE_FEATURES_LIST = SINGLE_FEATURES
        LSTM_FEATURES_LIST = LSTM_FEATURES
    elif FEATURES_MODE == "fm6":
        FEATURES_LIST = PRICE_FEATURES + TECH_INDICATORS #+ RETURNS_FEATURES
        SINGLE_FEATURES_LIST = SINGLE_FEATURES
        LSTM_FEATURES_LIST = LSTM_FEATURES
    elif FEATURES_MODE == "fm7":
        FEATURES_LIST = PRICE_FEATURES + RETURNS_FEATURES + TECH_INDICATORS + RISK_INDICATORS
        SINGLE_FEATURES_LIST = SINGLE_FEATURES
        LSTM_FEATURES_LIST = RETURNS_FEATURES + RISK_INDICATORS + SINGLE_FEATURES
    else:
        print("error (config): features list not found, cannot assign features mode.")
