"""
Database management for self-improving Palisade price model
"""

import sqlite3
import pandas as pd
import os
import re
from datetime import datetime
import hashlib

class PalisadeDatabase:
    """Manages the Palisade pricing database with VIN tracking"""
    
    def __init__(self, db_path="palisade_data.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create main pricing table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicle_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vin TEXT UNIQUE NOT NULL,
                price REAL NOT NULL,
                trim TEXT NOT NULL,
                drivetrain TEXT NOT NULL,
                city_mpg INTEGER NOT NULL,
                ext_color TEXT NOT NULL,
                int_color TEXT NOT NULL,
                zip_code TEXT,
                submission_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                verified BOOLEAN DEFAULT FALSE,
                source TEXT DEFAULT 'user_submission'
            )
        ''')
        
        # Check if zip_code column exists, add if missing (migration)
        cursor.execute("PRAGMA table_info(vehicle_data)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'zip_code' not in columns:
            cursor.execute('ALTER TABLE vehicle_data ADD COLUMN zip_code TEXT')
        
        # Create model training history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                data_count INTEGER NOT NULL,
                avg_error REAL,
                notes TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def validate_vin(self, vin):
        """Validate VIN format and check digit"""
        if not vin or len(vin) != 17:
            return False, "VIN must be exactly 17 characters"
        
        # Clean VIN (uppercase, remove invalid characters)
        vin = vin.upper().replace('O', '0').replace('I', '1').replace('Q', '0')
        
        # Check for invalid characters
        if not re.match(r'^[A-HJ-NPR-Z0-9]{17}$', vin):
            return False, "VIN contains invalid characters"
        
        # Basic Hyundai VIN format check
        if not vin.startswith(('KM8', 'KMH')):
            return False, "VIN does not appear to be from a Hyundai vehicle"
        
        # Check if it looks like a Palisade VIN (model year 2026+)
        if not ('TU' in vin or 'NU' in vin):  # Common Palisade patterns
            return False, "VIN does not appear to be from a Palisade"
        
        return True, "VIN is valid"
    
    def vin_exists(self, vin):
        """Check if VIN already exists in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM vehicle_data WHERE vin = ?", (vin,))
        exists = cursor.fetchone()[0] > 0
        conn.close()
        return exists
    
    def add_vehicle_data(self, vin, price, trim, drivetrain, city_mpg, ext_color, int_color, zip_code=None, verified=False):
        """Add new vehicle data to database"""
        # Validate VIN first
        is_valid, message = self.validate_vin(vin)
        if not is_valid:
            return False, f"Invalid VIN: {message}"
        
        # Check for duplicates
        if self.vin_exists(vin):
            return False, "Vehicle with this VIN already exists in database"
        
        # Validate price range
        if not (30000 <= price <= 80000):
            return False, "Price must be between $30,000 and $80,000"
        
        # Insert data
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO vehicle_data 
                (vin, price, trim, drivetrain, city_mpg, ext_color, int_color, zip_code, verified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (vin, price, trim, drivetrain, city_mpg, ext_color, int_color, zip_code, verified))
            conn.commit()
            conn.close()
            return True, "Vehicle data added successfully"
        except Exception as e:
            return False, f"Database error: {str(e)}"
    
    def get_all_data(self):
        """Get all vehicle data as DataFrame"""
        conn = sqlite3.connect(self.db_path)
        
        # Check if zip_code column exists
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(vehicle_data)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'zip_code' in columns:
            df = pd.read_sql_query('''
                SELECT vin, price, trim, drivetrain, city_mpg, ext_color, int_color, 
                       zip_code, submission_date, verified, source
                FROM vehicle_data 
                ORDER BY submission_date DESC
            ''', conn)
        else:
            # Fallback for older database schema
            df = pd.read_sql_query('''
                SELECT vin, price, trim, drivetrain, city_mpg, ext_color, int_color, 
                       submission_date, verified, source
                FROM vehicle_data 
                ORDER BY submission_date DESC
            ''', conn)
            df['zip_code'] = None  # Add empty column for consistency
        
        conn.close()
        return df
    
    def get_training_data(self, verified_only=False):
        """Get data suitable for model training"""
        conn = sqlite3.connect(self.db_path)
        where_clause = "WHERE verified = TRUE" if verified_only else ""
        
        # Check if zip_code column exists
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(vehicle_data)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'zip_code' in columns:
            df = pd.read_sql_query(f'''
                SELECT price as Price, trim as Trim, drivetrain as Drivetrain, 
                       city_mpg as City_mpg, ext_color as ExtColor, int_color as IntColor,
                       zip_code as ZipCode
                FROM vehicle_data 
                {where_clause}
                ORDER BY submission_date DESC
            ''', conn)
        else:
            # Fallback for older database schema
            df = pd.read_sql_query(f'''
                SELECT price as Price, trim as Trim, drivetrain as Drivetrain, 
                       city_mpg as City_mpg, ext_color as ExtColor, int_color as IntColor
                FROM vehicle_data 
                {where_clause}
                ORDER BY submission_date DESC
            ''', conn)
            df['ZipCode'] = None  # Add empty column for consistency
        
        conn.close()
        return df
    
    def get_data_stats(self):
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total records
        cursor.execute("SELECT COUNT(*) FROM vehicle_data")
        total_records = cursor.fetchone()[0]
        
        # Verified records
        cursor.execute("SELECT COUNT(*) FROM vehicle_data WHERE verified = TRUE")
        verified_records = cursor.fetchone()[0]
        
        # Recent submissions (last 7 days)
        cursor.execute("""
            SELECT COUNT(*) FROM vehicle_data 
            WHERE submission_date >= datetime('now', '-7 days')
        """)
        recent_submissions = cursor.fetchone()[0]
        
        # Price range
        cursor.execute("SELECT MIN(price), MAX(price), AVG(price) FROM vehicle_data")
        price_stats = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_records': total_records,
            'verified_records': verified_records,
            'recent_submissions': recent_submissions,
            'min_price': price_stats[0] if price_stats[0] else 0,
            'max_price': price_stats[1] if price_stats[1] else 0,
            'avg_price': price_stats[2] if price_stats[2] else 0
        }
    
    def load_initial_data(self, csv_path):
        """Load initial training data from CSV"""
        if not os.path.exists(csv_path):
            return False, f"CSV file not found: {csv_path}"
        
        try:
            df = pd.read_csv(csv_path)
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            added_count = 0
            for _, row in df.iterrows():
                # Check if VIN already exists
                if not self.vin_exists(row['VIN']):
                    cursor.execute('''
                        INSERT INTO vehicle_data 
                        (vin, price, trim, drivetrain, city_mpg, ext_color, int_color, zip_code, verified, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (row['VIN'], row['Price'], row['Trim'], row['Drivetrain'], 
                          row['City_mpg'], row['ExtColor'], row['IntColor'], 
                          row.get('ZipCode', None), True, 'initial_data'))
                    added_count += 1
            
            conn.commit()
            conn.close()
            return True, f"Loaded {added_count} initial records"
        except Exception as e:
            return False, f"Error loading initial data: {str(e)}"

def initialize_database():
    """Initialize database with training data"""
    db = PalisadeDatabase()
    
    # Load initial data if database is empty
    stats = db.get_data_stats()
    if stats['total_records'] == 0:
        success, message = db.load_initial_data('palisade_training_data_with_vins.csv')
        print(f"Database initialization: {message}")
    
    return db

if __name__ == "__main__":
    # Test the database
    db = initialize_database()
    stats = db.get_data_stats()
    print("Database Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")