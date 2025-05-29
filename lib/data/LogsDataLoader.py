import os
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence

class LogsDataLoader:
    def __init__(self, dataset_name, dir_path="./data", window_type=None):
        """Provides support for loading and processing the prefixes generated from logs.
        
        Args:
            dataset_name: str: Name of the dataset
            dir_path: str: Path to the dataset directory
            window_type: str: Type of window for batching (day, week, month, or None)
            train_test_ratio: float: Default ratio for splitting training and test data
        """
        self.dataset_name = dataset_name
        self.dir_path = f"{dir_path}/{dataset_name}/processed"
        self.window_type = window_type
        
        # Data attributes
        self.traces = None  # Full prefixes DataFrame
        self.max_case_length = 0  # Maximum prefix length (capped at 40)
        self.train_df = None  # Training prefixes
        self.test_df = None  # Testing prefixes
        self.attr_map = None  # Vocabulary mapping (based on training data only)
        
    def load_data(self):
        """Load the processed prefix data.
        
        Returns:
            DataFrame containing all prefixes sorted by timestamp
        """
        # Load the processed prefixes
        prefixes_file = f"{self.dir_path}/prefixes.csv"
        
        if not os.path.exists(prefixes_file):
            raise FileNotFoundError(f"Prefixes file not found at {prefixes_file}. Run data_processing.py first.")
        
        self.traces = pd.read_csv(prefixes_file)
        
        # Sort by timestamp
        self.traces["last_event_time"] = pd.to_datetime(self.traces["last_event_time"])
        self.traces = self.traces.sort_values(by="last_event_time").reset_index(drop=True)
        
        # Calculate max prefix length from the data
        raw_max_length = self._get_max_case_length(self.traces["prefix"])
        
        # Cap max_case_length at 40
        if raw_max_length > 40:
            print(f"Maximum prefix length {raw_max_length} exceeds 40, capping at 40")
            self.max_case_length = 40
        else:
            self.max_case_length = raw_max_length
            
        print(f"Loaded {len(self.traces)} traces with max length {self.max_case_length}")
        return self.traces
        
    def _extract_metadata(self, train_df):
        """Extract vocabulary from the training data only.
        
        Args:
            train_df: DataFrame containing training prefixes
        """
        # Create vocabulary with special tokens
        keys = ["[PAD]", "[UNK]"]
        
        # Get unique activities from prefixes and next_act in training data only
        prefix_activities = set()
        for prefix in train_df["prefix"].values:
            prefix_activities.update(prefix.split())
        
        next_activities = set(train_df["next_act"].unique())
        all_activities = list(prefix_activities.union(next_activities))
        
        # Add activities to keys
        keys.extend(all_activities)
        val = range(len(keys))
        
        # Create mappings for model
        self.attr_map = {
            "x_word_dict": dict(zip(keys, val)),  # Full vocabulary with special tokens
            "y_word_dict": dict(zip(all_activities, range(len(all_activities))))  # Just activities for output
        }
        
        # Save metadata
        os.makedirs(self.dir_path, exist_ok=True)
        with open(f"{self.dir_path}/metadata.json", "w") as f:
            json.dump(self.attr_map, f, indent=2)
        
    def split_train_test(self, train_test_ratio):
        """Split the prefixes into training and testing sets based on timestamps.
        Also creates vocabulary mapping based on training data only.
        
        Args:
            train_test_ratio: float: Ratio for splitting training and test data.
        
        Returns:
            tuple: (train_df, test_df, attr_map)
        """
        if self.traces is None:
            self.load_data()
           
        # Sort by timestamp
        sorted_traces = self.traces.sort_values(by="last_event_time").reset_index(drop=True)
        
        # Split based on timestamp
        split_idx = int(len(sorted_traces) * train_test_ratio)
        self.train_df = sorted_traces.iloc[:split_idx]
        self.test_df = sorted_traces.iloc[split_idx:]
        
        # Create vocabulary mapping based on training data only
        self._extract_metadata(self.train_df)
        
        # Apply fixed-length prefixes if max_case_length is less than actual prefix lengths
        # This truncates longer prefixes to max_case_length
        self.train_df = self._apply_fixed_length_prefixes(self.train_df, self.max_case_length)
        self.test_df = self._apply_fixed_length_prefixes(self.test_df, self.max_case_length)
        
        # Report train/test split timestamp
        split_time = self.train_df["last_event_time"].max()
        print(f"Split time: {split_time}")
        print(f"Training set: {len(self.train_df)} traces")
        print(f"Testing set: {len(self.test_df)} traces")
        
        # Save the split data
        self.train_df.to_csv(f"{self.dir_path}/next_activity_train.csv", index=False)
        self.test_df.to_csv(f"{self.dir_path}/next_activity_test.csv", index=False)
        
        return self.train_df, self.test_df, self.attr_map
    
    def _apply_fixed_length_prefixes(self, df, max_length):
        """Apply maximum length constraint to prefixes.
        
        Args:
            df: DataFrame with prefixes
            max_length: Maximum prefix length
            
        Returns:
            DataFrame with fixed-length prefixes
        """
        # Create a new DataFrame to store the results
        result_rows = []
        
        # Process each row
        for _, row in df.iterrows():
            # Extract prefix and modify if needed
            prefix_activities = row['prefix'].split()
            
            # Truncate prefix if needed
            if len(prefix_activities) > max_length:
                # Keep only the most recent max_length activities
                prefix_activities = prefix_activities[-max_length:]
                row = row.copy()
                row['prefix'] = ' '.join(prefix_activities)
            
            # Add row to results
            result_rows.append(row)
        
        # Create new DataFrame from results
        return pd.DataFrame(result_rows)
    
    def create_batches(self, df, window_type=None):
        """Create batches of data based on a time window.
        
        Args:
            df: DataFrame containing prefixes
            window_type: Type of time window (day, week, month, or None)
            
        Returns:
            dict: Dictionary of batches keyed by time period
        """
        if window_type is None:
            window_type = self.window_type
            
        if window_type is None:
            # Return the entire dataframe as a single batch
            return {"full": df}
            
        batches = {}
        
        if window_type == "day":
            # Group by day
            df["day"] = df["last_event_time"].dt.strftime("%Y-%m-%d")
            for day, group in df.groupby("day"):
                batches[day] = group.drop("day", axis=1)
                
        elif window_type == "week":
            # Group by ISO week
            df["week"] = df["last_event_time"].apply(
                lambda x: f"{x.year}/{x.isocalendar()[1]:02d}")
            for week, group in df.groupby("week"):
                batches[week] = group.drop("week", axis=1)
                
        elif window_type == "month":
            # Group by month
            df["month"] = df["last_event_time"].dt.strftime("%Y/%m")
            for month, group in df.groupby("month"):
                batches[month] = group.drop("month", axis=1)
                
        else:
            raise ValueError(f"Invalid window_type: {window_type}")
            
        print(f"Created {len(batches)} {window_type} batches")
        return batches
        
    def encode_traces(self, df, attr_map=None, max_case_length=None):
        """Encode traces for model input using the vocabulary mapping.
        
        Args:
            df: DataFrame containing prefixes
            attr_map: Vocabulary mapping from activities to indices
            max_case_length: Maximum prefix length for padding/truncation
            
        Returns:
            tuple: (encoded_x, encoded_y, sequence_lengths)
        """
        if attr_map is None:
            attr_map = self.attr_map
            
        if max_case_length is None:
            max_case_length = self.max_case_length
        
        x_word_dict = attr_map["x_word_dict"]
        y_word_dict = attr_map["y_word_dict"]
        
        # Get prefixes and next activities
        prefixes = df["prefix"].values
        next_acts = df["next_act"].values
        
        # Initialize empty lists for encoded data
        encoded_x = []  # Encoded prefixes
        seq_lengths = []  # Length of each sequence
        
        # Process each prefix
        for prefix in prefixes:
            # Split the prefix into individual activities
            activities = prefix.split()
            
            # Record sequence length
            seq_len = len(activities)
            seq_lengths.append(seq_len)
            
            # Encode each activity in the prefix
            encoded_prefix = []
            for activity in activities:
                # Use [UNK] token for unseen activities
                if activity in x_word_dict:
                    encoded_prefix.append(x_word_dict[activity])
                else:
                    encoded_prefix.append(x_word_dict["[UNK]"])
            
            # Pad or truncate to max_case_length
            if len(encoded_prefix) < max_case_length:
                # Pad with [PAD] token index
                pad_token = x_word_dict["[PAD]"]
                encoded_prefix.extend([pad_token] * (max_case_length - len(encoded_prefix)))
            elif len(encoded_prefix) > max_case_length:
                # Truncate to max_case_length (keep most recent)
                encoded_prefix = encoded_prefix[-max_case_length:]
                seq_lengths[-1] = max_case_length
                
            encoded_x.append(encoded_prefix)
            
        # Encode next activities (target)
        encoded_y = []
        for act in next_acts:
            if act in y_word_dict:
                encoded_y.append(y_word_dict[act])
            else:
                # For unseen next activities, add them to y_word_dict with new index!
                new_idx = len(y_word_dict)
                y_word_dict[act] = new_idx
                encoded_y.append(new_idx)
        
        # one-hot encode the encoded_y
        encoded_y = np.eye(len(y_word_dict))[encoded_y]

        # Convert to numpy arrays
        encoded_x = np.array(encoded_x, dtype=np.int64)
        encoded_y = np.array(encoded_y, dtype=np.int64)
        seq_lengths = np.array(seq_lengths, dtype=np.int64)
        
        # Return encoded data
        return encoded_x, encoded_y, seq_lengths
    
    def prepare_torch_data(self, encoded_x, encoded_y, seq_lengths, batch_size=32, shuffle=True):
        """Prepare data for PyTorch model including batching.
        
        Args:
            encoded_x: Encoded prefix sequences
            encoded_y: Encoded next activities (targets)
            seq_lengths: Length of each sequence
            batch_size: Size of batches for training
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader: PyTorch DataLoader with batched data
        """
        from torch.utils.data import TensorDataset, DataLoader
        
        # Convert to PyTorch tensors
        x_tensor = torch.tensor(encoded_x, dtype=torch.long)
        y_tensor = torch.tensor(encoded_y, dtype=torch.long)
        seq_lengths = torch.tensor(seq_lengths, dtype=torch.long)
        
        # Create dataset and dataloader
        dataset = TensorDataset(x_tensor, y_tensor, seq_lengths)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        return dataloader
    
    def _get_max_case_length(self, prefixes):
        """Calculate the maximum length of prefixes in tokens.
        
        Args:
            prefixes: Series of prefix strings
            
        Returns:
            int: Maximum length
        """
        max_length = 0
        for prefix in prefixes:
            length = len(prefix.split())
            if length > max_length:
                max_length = length
        return max_length 