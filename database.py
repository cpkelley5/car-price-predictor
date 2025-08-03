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
        
        # Create enhanced vehicle features table (from window sticker data)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS enhanced_vehicle_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vin TEXT UNIQUE NOT NULL,
                seats TEXT,
                engine TEXT,
                horsepower TEXT,
                base_msrp REAL,
                destination_charge REAL,
                total_msrp REAL,
                packages TEXT,
                options TEXT,
                sticker_url TEXT,
                parse_notes TEXT,
                scrape_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (vin) REFERENCES vehicle_data (vin)
            )
        ''')
        
        # Create normalized vehicle options table for individual option tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicle_options (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vin TEXT NOT NULL,
                
                -- Paint Options
                premium_paint BOOLEAN DEFAULT FALSE,
                premium_paint_cost REAL DEFAULT 0,
                paint_name TEXT,
                
                -- Floor Protection
                floor_mats BOOLEAN DEFAULT FALSE,
                floor_mats_cost REAL DEFAULT 0,
                
                -- Cargo Options
                cargo_net BOOLEAN DEFAULT FALSE,
                cargo_net_cost REAL DEFAULT 0,
                cargo_tray BOOLEAN DEFAULT FALSE,
                cargo_tray_cost REAL DEFAULT 0,
                cargo_cover BOOLEAN DEFAULT FALSE,
                cargo_cover_cost REAL DEFAULT 0,
                cargo_blocks BOOLEAN DEFAULT FALSE,
                cargo_blocks_cost REAL DEFAULT 0,
                
                -- Safety & Emergency
                first_aid_kit BOOLEAN DEFAULT FALSE,
                first_aid_kit_cost REAL DEFAULT 0,
                
                -- Weather Protection
                severe_weather_kit BOOLEAN DEFAULT FALSE,
                severe_weather_kit_cost REAL DEFAULT 0,
                
                -- Totals
                total_options_cost REAL DEFAULT 0,
                options_count INTEGER DEFAULT 0,
                
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (vin) REFERENCES vehicle_data (vin)
            )
        ''')
        
        # Create standard features table for trim-level feature tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS standard_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vin TEXT NOT NULL,
                
                -- Wheel & Exterior
                wheel_size TEXT,  -- "18", "20", "21"
                sunroof_type TEXT,  -- "Power", "Dual-Pane"
                
                -- Seating & Interior 
                seating_material TEXT,  -- "H-Tex", "Leather", "Nappa Leather"
                front_seat_ventilation BOOLEAN DEFAULT FALSE,
                seat_memory BOOLEAN DEFAULT FALSE,
                ergo_motion BOOLEAN DEFAULT FALSE,
                relaxation_seats BOOLEAN DEFAULT FALSE,
                
                -- Safety Features
                blind_spot_type TEXT,  -- "Warning", "Collision-Avoidance"
                parking_collision_avoidance BOOLEAN DEFAULT FALSE,
                parking_side_warning BOOLEAN DEFAULT FALSE,
                
                -- Technology
                audio_system TEXT,  -- "Standard", "Bose Premium"
                head_up_display BOOLEAN DEFAULT FALSE,
                front_rear_dashcam BOOLEAN DEFAULT FALSE,
                remote_smart_park BOOLEAN DEFAULT FALSE,
                homelink_mirror BOOLEAN DEFAULT FALSE,
                
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (vin) REFERENCES vehicle_data (vin)
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
    
    def add_enhanced_features(self, vin, seats=None, engine=None, horsepower=None, 
                             base_msrp=None, destination_charge=None, total_msrp=None,
                             packages=None, options=None, sticker_url=None, parse_notes=None):
        """Add enhanced vehicle features from window sticker data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO enhanced_vehicle_features 
                (vin, seats, engine, horsepower, base_msrp, destination_charge, 
                 total_msrp, packages, options, sticker_url, parse_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (vin, seats, engine, horsepower, base_msrp, destination_charge,
                  total_msrp, packages, options, sticker_url, parse_notes))
            conn.commit()
            conn.close()
            return True, "Enhanced features added successfully"
        except Exception as e:
            return False, f"Database error: {str(e)}"
    
    def get_enhanced_features(self, vin=None):
        """Get enhanced features data"""
        conn = sqlite3.connect(self.db_path)
        if vin:
            df = pd.read_sql_query('''
                SELECT * FROM enhanced_vehicle_features WHERE vin = ?
            ''', conn, params=(vin,))
        else:
            df = pd.read_sql_query('''
                SELECT * FROM enhanced_vehicle_features ORDER BY scrape_date DESC
            ''', conn)
        conn.close()
        return df
    
    def add_vehicle_options(self, vin, **options):
        """Add normalized vehicle options to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate totals
            total_cost = sum(v for k, v in options.items() if k.endswith('_cost') and v)
            options_count = sum(1 for k, v in options.items() if k.endswith('_cost') and v > 0)
            
            # Prepare column names and values
            columns = list(options.keys()) + ['total_options_cost', 'options_count']
            values = list(options.values()) + [total_cost, options_count]
            placeholders = ', '.join(['?'] * len(columns))
            column_names = ', '.join(columns)
            
            cursor.execute(f'''
                INSERT OR REPLACE INTO vehicle_options 
                (vin, {column_names})
                VALUES (?, {placeholders})
            ''', [vin] + values)
            
            conn.commit()
            conn.close()
            return True, "Vehicle options added successfully"
        except Exception as e:
            return False, f"Database error: {str(e)}"
    
    def get_vehicle_options(self, vin=None):
        """Get normalized vehicle options data"""
        conn = sqlite3.connect(self.db_path)
        if vin:
            df = pd.read_sql_query('''
                SELECT * FROM vehicle_options WHERE vin = ?
            ''', conn, params=(vin,))
        else:
            df = pd.read_sql_query('''
                SELECT * FROM vehicle_options ORDER BY created_date DESC
            ''', conn)
        conn.close()
        return df
    
    def add_standard_features(self, vin, **features):
        """Add standard features based on trim level"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Prepare column names and values
            columns = list(features.keys())
            values = list(features.values())
            placeholders = ', '.join(['?'] * len(columns))
            column_names = ', '.join(columns)
            
            cursor.execute(f'''
                INSERT OR REPLACE INTO standard_features 
                (vin, {column_names})
                VALUES (?, {placeholders})
            ''', [vin] + values)
            
            conn.commit()
            conn.close()
            return True, "Standard features added successfully"
        except Exception as e:
            return False, f"Database error: {str(e)}"
    
    def get_standard_features(self, vin=None):
        """Get standard features data"""
        conn = sqlite3.connect(self.db_path)
        if vin:
            df = pd.read_sql_query('''
                SELECT * FROM standard_features WHERE vin = ?
            ''', conn, params=(vin,))
        else:
            df = pd.read_sql_query('''
                SELECT * FROM standard_features ORDER BY created_date DESC
            ''', conn)
        conn.close()
        return df

    def get_training_data_with_options(self, verified_only=False):
        """Get training data enhanced with normalized option and standard feature data"""
        conn = sqlite3.connect(self.db_path)
        where_clause = "WHERE vd.verified = TRUE" if verified_only else ""
        
        df = pd.read_sql_query(f'''
            SELECT 
                vd.price as Price, 
                vd.trim as Trim, 
                vd.drivetrain as Drivetrain,
                vd.city_mpg as City_mpg, 
                vd.ext_color as ExtColor, 
                vd.int_color as IntColor,
                vd.zip_code as ZipCode,
                
                -- Options
                COALESCE(vo.premium_paint, 0) as PremiumPaint,
                COALESCE(vo.floor_mats, 0) as FloorMats,
                COALESCE(vo.cargo_net, 0) as CargoNet,
                COALESCE(vo.cargo_tray, 0) as CargoTray,
                COALESCE(vo.cargo_cover, 0) as CargoCover,
                COALESCE(vo.cargo_blocks, 0) as CargoBlocks,
                COALESCE(vo.first_aid_kit, 0) as FirstAidKit,
                COALESCE(vo.severe_weather_kit, 0) as SevereWeatherKit,
                COALESCE(vo.total_options_cost, 0) as TotalOptionsCost,
                COALESCE(vo.options_count, 0) as OptionsCount,
                
                -- Standard Features
                sf.wheel_size as WheelSize,
                sf.sunroof_type as SunroofType,
                sf.seating_material as SeatingMaterial,
                COALESCE(sf.front_seat_ventilation, 0) as FrontSeatVentilation,
                COALESCE(sf.seat_memory, 0) as SeatMemory,
                COALESCE(sf.ergo_motion, 0) as ErgoMotion,
                COALESCE(sf.relaxation_seats, 0) as RelaxationSeats,
                sf.blind_spot_type as BlindSpotType,
                COALESCE(sf.parking_collision_avoidance, 0) as ParkingCollisionAvoidance,
                COALESCE(sf.parking_side_warning, 0) as ParkingSideWarning,
                sf.audio_system as AudioSystem,
                COALESCE(sf.head_up_display, 0) as HeadUpDisplay,
                COALESCE(sf.front_rear_dashcam, 0) as FrontRearDashcam,
                COALESCE(sf.remote_smart_park, 0) as RemoteSmartPark,
                COALESCE(sf.homelink_mirror, 0) as HomelinkMirror
                
            FROM vehicle_data vd
            LEFT JOIN vehicle_options vo ON vd.vin = vo.vin
            LEFT JOIN standard_features sf ON vd.vin = sf.vin
            {where_clause}
            ORDER BY vd.submission_date DESC
        ''', conn)
        
        conn.close()
        return df

    def get_vins_without_enhanced_data(self):
        """Get VINs that don't have enhanced feature data yet"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT vd.vin, vd.trim, vd.price 
            FROM vehicle_data vd
            LEFT JOIN enhanced_vehicle_features evf ON vd.vin = evf.vin
            WHERE evf.vin IS NULL
            ORDER BY vd.submission_date DESC
        ''', conn)
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