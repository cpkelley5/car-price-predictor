#!/usr/bin/env python3
"""
Production Database Initialization Script
Ensures all required tables exist in production environment
"""

from database import PalisadeDatabase
import os

def init_production_database():
    """Initialize production database with all required tables"""
    print("🔧 Initializing production database...")
    
    # Initialize database (this will create all tables)
    db = PalisadeDatabase()
    
    # Check if basic data exists
    stats = db.get_data_stats()
    print(f"📊 Database stats: {stats}")
    
    # Load initial data if needed
    if stats['total_records'] == 0:
        csv_path = 'palisade_training_data_with_vins.csv'
        if os.path.exists(csv_path):
            success, message = db.load_initial_data(csv_path)
            print(f"💾 Initial data loading: {message}")
        else:
            print(f"⚠️  Warning: Training data file {csv_path} not found")
    
    # Verify enhanced features table exists
    try:
        enhanced_data = db.get_enhanced_features()
        print(f"✅ Enhanced features table exists with {len(enhanced_data)} records")
    except Exception as e:
        print(f"❌ Enhanced features table error: {e}")
    
    # Test enhanced features creation
    try:
        # This will create the table if it doesn't exist
        success, message = db.add_enhanced_features(
            vin="TEST_VIN_DELETE_ME",
            parse_notes="Test initialization record"
        )
        if success:
            print("✅ Enhanced features table creation test passed")
            # Clean up test record
            import sqlite3
            conn = sqlite3.connect(db.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM enhanced_vehicle_features WHERE vin = 'TEST_VIN_DELETE_ME'")
            conn.commit()
            conn.close()
            print("🧹 Cleaned up test record")
        else:
            print(f"❌ Enhanced features test failed: {message}")
    except Exception as e:
        print(f"❌ Enhanced features creation test error: {e}")
    
    print("🎉 Production database initialization complete!")

if __name__ == "__main__":
    init_production_database()