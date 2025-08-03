"""
Integration module for window sticker scraper with our database system
"""

import sys
import os
from typing import List, Dict, Optional

try:
    import pandas as pd
    from database import PalisadeDatabase
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Add sticker-scraper directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'sticker-scraper'))

try:
    from sticker_scraper import fetch_sticker_pdf, pdf_to_text, parse_sticker_text, StickerData
    SCRAPER_AVAILABLE = True
except ImportError:
    SCRAPER_AVAILABLE = False

class StickerDataEnhancer:
    """Enhances our vehicle database with window sticker data"""
    
    def __init__(self, db_path="palisade_data.db"):
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas and database module required")
        self.db = PalisadeDatabase(db_path)
        
    def check_scraper_availability(self):
        """Check if scraper dependencies are available"""
        if not SCRAPER_AVAILABLE:
            return False, "Scraper module not available. Check sticker-scraper directory."
        
        try:
            import requests
            import pdfplumber
            return True, "Scraper ready"
        except ImportError as e:
            return False, f"Missing dependencies: {e}"
    
    def enhance_single_vin(self, vin: str) -> tuple[bool, str, Optional[StickerData]]:
        """Enhance a single VIN with window sticker data"""
        if not SCRAPER_AVAILABLE:
            return False, "Scraper not available", None
        
        try:
            # Fetch and parse sticker
            pdf_bytes = fetch_sticker_pdf(vin)
            text = pdf_to_text(pdf_bytes)
            
            if not text or len(text.strip()) < 50:
                sticker_data = StickerData(
                    VIN=vin, 
                    ParseNotes="Sticker text extraction empty; may be scanned image"
                )
            else:
                sticker_data = parse_sticker_text(vin, text)
            
            # Store in database
            success, message = self.db.add_enhanced_features(
                vin=sticker_data.VIN,
                seats=sticker_data.Seats,
                engine=sticker_data.Engine,
                horsepower=sticker_data.Horsepower,
                base_msrp=sticker_data.BaseMSRP,
                destination_charge=sticker_data.DestinationCharge,
                total_msrp=sticker_data.TotalMSRP,
                packages=sticker_data.Packages,
                options=sticker_data.Options,
                sticker_url=sticker_data.StickerURL,
                parse_notes=sticker_data.ParseNotes
            )
            
            if success:
                return True, f"Enhanced VIN {vin} successfully", sticker_data
            else:
                return False, f"Database error: {message}", sticker_data
                
        except Exception as e:
            return False, f"Error enhancing VIN {vin}: {str(e)}", None
    
    def enhance_multiple_vins(self, vins: List[str], delay_sec: float = 0.5) -> Dict[str, tuple[bool, str]]:
        """Enhance multiple VINs with progress tracking"""
        import time
        
        results = {}
        for vin in vins:
            vin = vin.strip()
            if not vin:
                continue
                
            success, message, _ = self.enhance_single_vin(vin)
            results[vin] = (success, message)
            
            # Be polite to the server
            time.sleep(delay_sec)
        
        return results
    
    def get_enhancement_candidates(self) -> pd.DataFrame:
        """Get VINs that could benefit from enhancement"""
        return self.db.get_vins_without_enhanced_data()
    
    def get_enhancement_stats(self) -> Dict[str, int]:
        """Get statistics about enhanced data"""
        all_vins = len(self.db.get_all_data())
        enhanced_vins = len(self.db.get_enhanced_features())
        pending_vins = len(self.get_enhancement_candidates())
        
        return {
            'total_vins': all_vins,
            'enhanced_vins': enhanced_vins,
            'pending_vins': pending_vins,
            'enhancement_rate': enhanced_vins / all_vins if all_vins > 0 else 0
        }
    
    def get_enhanced_summary(self) -> pd.DataFrame:
        """Get summary of enhanced vehicle data"""
        enhanced_df = self.db.get_enhanced_features()
        if enhanced_df.empty:
            return enhanced_df
        
        # Add some computed fields for analysis
        enhanced_df['has_packages'] = enhanced_df['packages'].notna()
        enhanced_df['has_options'] = enhanced_df['options'].notna() 
        enhanced_df['parse_success'] = enhanced_df['parse_notes'].isna()
        
        return enhanced_df

if __name__ == "__main__":
    # Test the enhancer
    enhancer = StickerDataEnhancer()
    
    # Check availability
    available, message = enhancer.check_scraper_availability()
    print(f"Scraper available: {available} - {message}")
    
    # Get stats
    stats = enhancer.get_enhancement_stats()
    print(f"Enhancement stats: {stats}")
    
    # Get candidates
    candidates = enhancer.get_enhancement_candidates()
    print(f"VINs ready for enhancement: {len(candidates)}")