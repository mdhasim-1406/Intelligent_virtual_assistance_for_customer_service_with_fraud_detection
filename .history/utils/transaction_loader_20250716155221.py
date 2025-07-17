"""
SafeServe AI - Transaction Loader
Utility to load and process transaction data from various sources
"""

import pandas as pd
import json
import csv
from typing import List, Dict, Any, Optional
import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionLoader:
    """
    Utility class for loading and processing transaction data
    """
    
    def __init__(self):
        self.supported_formats = ['.csv', '.json', '.xlsx']
    
    def load_csv(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load transactions from CSV file
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            List of transaction dictionaries
        """
        try:
            df = pd.read_csv(filepath)
            transactions = df.to_dict('records')
            logger.info(f"Loaded {len(transactions)} transactions from CSV: {filepath}")
            return transactions
        except Exception as e:
            logger.error(f"Error loading CSV file {filepath}: {str(e)}")
            raise
    
    def load_json(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load transactions from JSON file
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            List of transaction dictionaries
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                transactions = data
            elif isinstance(data, dict) and 'transactions' in data:
                transactions = data['transactions']
            else:
                transactions = [data]
            
            logger.info(f"Loaded {len(transactions)} transactions from JSON: {filepath}")
            return transactions
        except Exception as e:
            logger.error(f"Error loading JSON file {filepath}: {str(e)}")
            raise
    
    def load_excel(self, filepath: str, sheet_name: str = 0) -> List[Dict[str, Any]]:
        """
        Load transactions from Excel file
        
        Args:
            filepath: Path to Excel file
            sheet_name: Sheet name or index
            
        Returns:
            List of transaction dictionaries
        """
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
            transactions = df.to_dict('records')
            logger.info(f"Loaded {len(transactions)} transactions from Excel: {filepath}")
            return transactions
        except Exception as e:
            logger.error(f"Error loading Excel file {filepath}: {str(e)}")
            raise
    
    def load_transactions(self, filepath: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Load transactions from file (auto-detect format)
        
        Args:
            filepath: Path to transaction file
            **kwargs: Additional arguments for specific loaders
            
        Returns:
            List of transaction dictionaries
        """
        file_path = Path(filepath)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Transaction file not found: {filepath}")
        
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.csv':
            return self.load_csv(filepath)
        elif file_ext == '.json':
            return self.load_json(filepath)
        elif file_ext in ['.xlsx', '.xls']:
            return self.load_excel(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: {self.supported_formats}")
    
    def normalize_transactions(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize transaction data to standard format
        
        Args:
            transactions: List of raw transaction dictionaries
            
        Returns:
            List of normalized transaction dictionaries
        """
        normalized = []
        
        for transaction in transactions:
            normalized_txn = self._normalize_single_transaction(transaction)
            if normalized_txn:
                normalized.append(normalized_txn)
        
        logger.info(f"Normalized {len(normalized)} transactions")
        return normalized
    
    def _normalize_single_transaction(self, transaction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Normalize a single transaction to standard format
        
        Args:
            transaction: Raw transaction dictionary
            
        Returns:
            Normalized transaction dictionary or None if invalid
        """
        try:
            # Standard field mapping
            field_mapping = {
                'amount': ['amount', 'value', 'transaction_amount', 'amt'],
                'timestamp': ['timestamp', 'date', 'transaction_date', 'time'],
                'location': ['location', 'place', 'city', 'country'],
                'device_id': ['device_id', 'device', 'device_fingerprint'],
                'merchant': ['merchant', 'merchant_name', 'shop', 'store'],
                'user_id': ['user_id', 'customer_id', 'account_id'],
                'velocity_score': ['velocity_score', 'frequency', 'velocity']
            }
            
            normalized = {}
            
            # Map fields
            for standard_field, possible_fields in field_mapping.items():
                value = None
                for field in possible_fields:
                    if field in transaction:
                        value = transaction[field]
                        break
                
                if standard_field == 'amount':
                    # Required field
                    if value is None:
                        logger.warning(f"Missing required field 'amount' in transaction: {transaction}")
                        return None
                    normalized[standard_field] = float(value)
                elif standard_field == 'timestamp':
                    if value:
                        normalized[standard_field] = self._parse_timestamp(value)
                    else:
                        normalized[standard_field] = datetime.datetime.now().isoformat()
                elif standard_field == 'velocity_score':
                    normalized[standard_field] = float(value) if value is not None else 0.0
                else:
                    normalized[standard_field] = str(value) if value is not None else 'unknown'
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing transaction {transaction}: {str(e)}")
            return None
    
    def _parse_timestamp(self, timestamp_value: Any) -> str:
        """
        Parse various timestamp formats to ISO format
        
        Args:
            timestamp_value: Timestamp in various formats
            
        Returns:
            ISO formatted timestamp string
        """
        if isinstance(timestamp_value, str):
            # Try parsing common formats
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d',
                '%d/%m/%Y %H:%M:%S',
                '%d/%m/%Y',
                '%m/%d/%Y %H:%M:%S',
                '%m/%d/%Y'
            ]
            
            for fmt in formats:
                try:
                    return datetime.datetime.strptime(timestamp_value, fmt).isoformat()
                except ValueError:
                    continue
            
            # If no format matches, try pandas parsing
            try:
                return pd.to_datetime(timestamp_value).isoformat()
            except:
                logger.warning(f"Could not parse timestamp: {timestamp_value}")
                return datetime.datetime.now().isoformat()
        
        elif isinstance(timestamp_value, (int, float)):
            # Assume Unix timestamp
            return datetime.datetime.fromtimestamp(timestamp_value).isoformat()
        
        else:
            return datetime.datetime.now().isoformat()
    
    def save_transactions(self, transactions: List[Dict[str, Any]], filepath: str):
        """
        Save transactions to file
        
        Args:
            transactions: List of transaction dictionaries
            filepath: Output file path
        """
        file_path = Path(filepath)
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.csv':
            df = pd.DataFrame(transactions)
            df.to_csv(filepath, index=False)
        elif file_ext == '.json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(transactions, f, indent=2, default=str)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.DataFrame(transactions)
            df.to_excel(filepath, index=False)
        else:
            raise ValueError(f"Unsupported output format: {file_ext}")
        
        logger.info(f"Saved {len(transactions)} transactions to {filepath}")
    
    def generate_sample_transactions(self, count: int = 100) -> List[Dict[str, Any]]:
        """
        Generate sample transaction data for testing
        
        Args:
            count: Number of sample transactions to generate
            
        Returns:
            List of sample transaction dictionaries
        """
        import random
        import numpy as np
        
        np.random.seed(42)
        random.seed(42)
        
        merchants = ['Amazon', 'Walmart', 'Target', 'Starbucks', 'McDonald\'s', 'Gas Station', 'ATM', 'Online Shop']
        locations = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'foreign', 'unknown']
        
        transactions = []
        
        for i in range(count):
            # Generate realistic transaction data
            amount = round(np.random.lognormal(mean=3, sigma=1), 2)
            timestamp = datetime.datetime.now() - datetime.timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            transaction = {
                'amount': amount,
                'timestamp': timestamp.isoformat(),
                'location': random.choice(locations),
                'device_id': f'device_{random.randint(1000, 9999)}',
                'merchant': random.choice(merchants),
                'user_id': f'user_{random.randint(1000, 9999)}',
                'velocity_score': round(np.random.exponential(scale=2), 2)
            }
            
            transactions.append(transaction)
        
        logger.info(f"Generated {count} sample transactions")
        return transactions

# Example usage
if __name__ == "__main__":
    loader = TransactionLoader()
    
    # Generate sample data
    sample_transactions = loader.generate_sample_transactions(50)
    
    # Save as different formats
    loader.save_transactions(sample_transactions, "sample_transactions.csv")
    loader.save_transactions(sample_transactions, "sample_transactions.json")
    
    # Load and normalize
    loaded_transactions = loader.load_transactions("sample_transactions.csv")
    normalized_transactions = loader.normalize_transactions(loaded_transactions)
    
    print(f"Generated and processed {len(normalized_transactions)} sample transactions")
    print("Sample transaction:", normalized_transactions[0] if normalized_transactions else "None")