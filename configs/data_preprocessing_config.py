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
