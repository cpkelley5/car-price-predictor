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
    from sticker_scraper import fetch_sticker_pdf, fetch_sticker_pdf_browser, pdf_to_text, parse_sticker_text, StickerData, SELENIUM_AVAILABLE
    SCRAPER_AVAILABLE = True
except ImportError:
    SCRAPER_AVAILABLE = False
    SELENIUM_AVAILABLE = False

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
            
            # Check for browser automation
            browser_available = SELENIUM_AVAILABLE
            if browser_available:
                return True, "Scraper ready (with browser automation support)"
            else:
                return True, "Scraper ready (requests-only mode)"
        except ImportError as e:
            return False, f"Missing dependencies: {e}"
    
    def test_browser_availability(self):
        """Test if browser automation actually works"""
        if not SELENIUM_AVAILABLE:
            return False, "Selenium not available"
        
        try:
            from sticker_scraper import create_chrome_driver
            test_driver = create_chrome_driver(headless=True)
            test_driver.quit()
            return True, "Browser automation working"
        except Exception as e:
            return False, f"ChromeDriver unavailable: {str(e)[:100]}..."
    
    def enhance_single_vin(self, vin: str, use_browser: bool = False) -> tuple[bool, str, Optional[StickerData]]:
        """Enhance a single VIN with window sticker data"""
        if not SCRAPER_AVAILABLE:
            return False, "Scraper not available", None
        
        try:
            # Try browser mode first if available and requested, then fallback to requests
            pdf_bytes = None
            method_used = ""
            
            if use_browser and SELENIUM_AVAILABLE:
                try:
                    pdf_bytes = fetch_sticker_pdf_browser(vin)
                    method_used = "browser"
                except Exception as browser_error:
                    # Fallback to requests if browser fails
                    try:
                        pdf_bytes = fetch_sticker_pdf(vin)
                        method_used = "requests (browser fallback)"
                    except Exception:
                        raise browser_error  # Report the original browser error
            else:
                try:
                    pdf_bytes = fetch_sticker_pdf(vin)
                    method_used = "requests"
                except Exception as requests_error:
                    # Try browser as fallback if available
                    if SELENIUM_AVAILABLE:
                        pdf_bytes = fetch_sticker_pdf_browser(vin)
                        method_used = "browser (requests fallback)"
                    else:
                        raise requests_error
            
            text = pdf_to_text(pdf_bytes)
            
            if not text or len(text.strip()) < 50:
                sticker_data = StickerData(
                    VIN=vin, 
                    ParseNotes=f"Sticker text extraction empty; may be scanned image (method: {method_used})"
                )
            else:
                sticker_data = parse_sticker_text(vin, text)
                # Add method used to parse notes
                if sticker_data.ParseNotes:
                    sticker_data.ParseNotes += f" (method: {method_used})"
                else:
                    sticker_data.ParseNotes = f"Successfully parsed via {method_used}"
            
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
                return True, f"Enhanced VIN {vin} successfully via {method_used}", sticker_data
            else:
                return False, f"Database error: {message}", sticker_data
                
        except Exception as e:
            return False, f"Error enhancing VIN {vin}: {str(e)}", None
    
    def enhance_multiple_vins(self, vins: List[str], delay_sec: float = 2.0) -> Dict[str, tuple[bool, str]]:
        """Enhance multiple VINs with progress tracking and respectful delays"""
        import time
        import random
        
        results = {}
        for i, vin in enumerate(vins):
            vin = vin.strip()
            if not vin:
                continue
                
            try:
                success, message, _ = self.enhance_single_vin(vin)
                results[vin] = (success, message)
                
                # Be very polite to the server - randomized delays to look more human
                if i < len(vins) - 1:  # Don't delay after the last VIN
                    actual_delay = delay_sec + random.uniform(0.5, 2.0)  # 2.5-4 second range
                    time.sleep(actual_delay)
                    
            except Exception as e:
                results[vin] = (False, f"Unexpected error: {str(e)}")
                # If we hit errors, wait even longer
                if "403" in str(e) or "Forbidden" in str(e):
                    time.sleep(10)  # Wait 10 seconds on 403 errors
        
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