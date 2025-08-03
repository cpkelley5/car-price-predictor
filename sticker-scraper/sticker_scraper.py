#!/usr/bin/env python3
"""
Hyundai Window Sticker Scraper (College Park Hyundai)
-----------------------------------------------------

What it does
- Given one or more VINs, downloads the window sticker PDF from College Park Hyundai:
  https://www.collegeparkhyundai.com/dealer-inspire-inventory/window-stickers/hyundai/?vin={VIN}
- Parses the PDF text to extract critical fields for pricing models:
  VIN, Trim, Drivetrain, Seats, Engine, Horsepower, ExteriorColor, InteriorColor,
  BaseMSRP, DestinationCharge, TotalMSRP, Packages, Options.
- Writes a normalized CSV.

Usage
- Single VIN:
    python sticker_scraper.py --vin KM8RM5S22TU019312 --out stickers.csv

- Many VINs from a text file (one VIN per line):
    python sticker_scraper.py --vin-file vins.txt --out stickers.csv

Notes
- Window sticker layouts vary by dealership and model year. The regex rules here
  work for the most common Hyundai sticker formats, but you may need to tweak patterns.
- If a sticker is scanned as an image instead of text, OCR would be required. See the
  --enable-ocr option note below if you want to add pytesseract support later.
"""

import argparse
import io
import re
import sys
import time
import csv
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

import requests
import pdfplumber

# Optional browser automation support
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# Optional future OCR support (disabled by default)
# import pytesseract
# from pdf2image import convert_from_bytes

COLLEGE_PARK_STICKER_URL = "https://www.collegeparkhyundai.com/dealer-inspire-inventory/window-stickers/hyundai/?vin={vin}"

def create_chrome_driver(headless=True, download_dir=None):
    """Create a Chrome WebDriver with realistic settings"""
    if not SELENIUM_AVAILABLE:
        raise ImportError("Selenium not available. Install with: pip install selenium")
    
    options = Options()
    if headless:
        options.add_argument("--headless")
    
    # Configure download directory and behavior
    if download_dir:
        import os
        os.makedirs(download_dir, exist_ok=True)
        prefs = {
            "download.default_directory": download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            "plugins.always_open_pdf_externally": True  # Download PDFs instead of viewing
        }
        options.add_experimental_option("prefs", prefs)
    
    # Make browser look more realistic
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-crash-reporter")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-in-process-stack-traces")
    options.add_argument("--disable-logging")
    options.add_argument("--disable-dev-tools")
    options.add_argument("--log-level=3")
    options.add_argument("--output=/dev/null")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    # Try different approaches for driver creation
    driver_errors = []
    
    # Method 1: Try with ChromeDriverManager (if available)
    try:
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        # Execute script to remove webdriver property
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        return driver
    except Exception as e:
        driver_errors.append(f"ChromeDriverManager failed: {e}")
    
    # Method 2: Try with default Chrome path
    try:
        driver = webdriver.Chrome(options=options)
        # Execute script to remove webdriver property  
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        return driver
    except Exception as e:
        driver_errors.append(f"Default Chrome failed: {e}")
    
    # Method 3: Try with explicit Chrome binary paths for cloud environments
    chrome_paths = [
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable", 
        "/usr/bin/chromium-browser",
        "/usr/bin/chromium"
    ]
    
    for chrome_path in chrome_paths:
        try:
            import os
            if os.path.exists(chrome_path):
                options.binary_location = chrome_path
                driver = webdriver.Chrome(options=options)
                driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
                return driver
        except Exception as e:
            driver_errors.append(f"Chrome path {chrome_path} failed: {e}")
    
    # If all methods fail, raise comprehensive error
    error_msg = "Failed to create Chrome driver. Tried multiple methods:\n" + "\n".join(driver_errors)
    error_msg += "\n\nThis usually means ChromeDriver is not available in this environment."
    raise WebDriverException(error_msg)

def fetch_sticker_pdf_browser(vin: str, timeout: int = 30) -> bytes:
    """Fetch sticker PDF using real browser to bypass bot detection"""
    if not SELENIUM_AVAILABLE:
        raise ImportError("Browser mode requires selenium. Install with: pip install selenium")
    
    import tempfile
    import os
    import glob
    
    url = COLLEGE_PARK_STICKER_URL.format(vin=vin.strip())
    driver = None
    
    try:
        # Create temporary download directory
        download_dir = tempfile.mkdtemp()
        driver = create_chrome_driver(headless=True, download_dir=download_dir)
        driver.set_page_load_timeout(timeout)
        
        # Navigate to the URL
        driver.get(url)
        
        # Wait a bit to let any JS load
        time.sleep(3)
        
        # Look for and click the "Window Sticker" button
        try:
            # Try multiple selectors for the Window Sticker button
            button_selectors = [
                "//button[contains(text(), 'Window Sticker')]",
                "//a[contains(text(), 'Window Sticker')]", 
                "//button[contains(@class, 'window-sticker')]",
                "//a[contains(@class, 'window-sticker')]",
                "//*[contains(text(), 'Window Sticker') and (self::button or self::a)]"
            ]
            
            button_found = False
            for selector in button_selectors:
                try:
                    button = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                    button.click()
                    button_found = True
                    print(f"Clicked Window Sticker button for VIN {vin}")
                    break
                except TimeoutException:
                    continue
            
            if button_found:
                # Wait for PDF to download
                print(f"Waiting for PDF download for VIN {vin}...")
                
                # Monitor download directory for PDF file
                download_timeout = 30
                check_interval = 1
                elapsed = 0
                
                while elapsed < download_timeout:
                    time.sleep(check_interval)
                    elapsed += check_interval
                    
                    # Look for PDF files in download directory
                    pdf_files = glob.glob(os.path.join(download_dir, "*.pdf"))
                    if pdf_files:
                        # Found a PDF, read it
                        pdf_path = pdf_files[0]  # Take the first PDF found
                        with open(pdf_path, 'rb') as f:
                            pdf_content = f.read()
                        
                        # Cleanup
                        try:
                            os.remove(pdf_path)
                            os.rmdir(download_dir)
                        except:
                            pass
                        
                        print(f"Successfully downloaded PDF for VIN {vin}")
                        return pdf_content
                
                # If we get here, download timed out
                raise TimeoutException(f"PDF download timed out for VIN {vin}")
            else:
                # Button not found, check page for errors
                page_source = driver.page_source.lower()
                if "403" in page_source or "forbidden" in page_source:
                    raise requests.exceptions.HTTPError(f"Access denied for VIN {vin}")
                elif "not found" in page_source or "404" in page_source:
                    raise requests.exceptions.HTTPError(f"Window sticker not found for VIN {vin}")
                elif "invalid" in page_source and "vin" in page_source:
                    raise requests.exceptions.HTTPError(f"Invalid VIN {vin} according to dealership")
                else:
                    raise ValueError(f"Could not find Window Sticker button for VIN {vin}")
                
        except Exception as e:
            print(f"Error during browser automation for VIN {vin}: {e}")
            raise
                
    finally:
        if driver:
            driver.quit()
        
        # Cleanup download directory
        try:
            import shutil
            if 'download_dir' in locals() and os.path.exists(download_dir):
                shutil.rmtree(download_dir)
        except:
            pass

# ------------------------------
# Helpers
# ------------------------------

def fetch_sticker_pdf(vin: str, timeout: int = 30) -> bytes:
    url = COLLEGE_PARK_STICKER_URL.format(vin=vin.strip())
    
    # More realistic headers to avoid bot detection
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0"
    }
    
    # Add session for cookie persistence
    session = requests.Session()
    session.headers.update(headers)
    
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        
        if "application/pdf" not in r.headers.get("Content-Type", ""):
            # Some sites return PDF without proper header; still try to parse if bytes look like PDF
            if not r.content.startswith(b"%PDF"):
                # Check if we got an error page instead
                if b"403" in r.content or b"Forbidden" in r.content:
                    raise requests.exceptions.HTTPError(f"Access denied for VIN {vin}. Dealership may be blocking automated requests.")
                raise ValueError(f"Sticker for VIN {vin} did not return a PDF. Content-Type: {r.headers.get('Content-Type')}")
        
        return r.content
        
    except requests.exceptions.HTTPError as e:
        if e.response and e.response.status_code == 403:
            raise requests.exceptions.HTTPError(f"Access denied for VIN {vin}. Try again later or the dealership may be blocking automated requests.")
        raise


def pdf_to_text(pdf_bytes: bytes) -> str:
    """Extract text from a PDF using pdfplumber."""
    text_chunks = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text:
                text_chunks.append(text)
    return "\n".join(text_chunks)


def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class StickerData:
    VIN: str
    Trim: Optional[str] = None
    Drivetrain: Optional[str] = None
    Seats: Optional[str] = None
    Engine: Optional[str] = None
    Horsepower: Optional[str] = None
    ExteriorColor: Optional[str] = None
    InteriorColor: Optional[str] = None
    BaseMSRP: Optional[float] = None
    DestinationCharge: Optional[float] = None
    TotalMSRP: Optional[float] = None
    Packages: Optional[str] = None
    Options: Optional[str] = None
    StickerURL: Optional[str] = None
    ParseNotes: Optional[str] = None


CURRENCY_RE = r"\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)"

def parse_currency(val: Optional[str]) -> Optional[float]:
    if not val:
        return None
    m = re.search(CURRENCY_RE, val.replace(",", ""))
    try:
        # Remove commas first, then cast
        cleaned = re.sub(r"[^\d.]", "", val)
        return float(cleaned) if cleaned else None
    except Exception:
        return None


def extract_with_regex(text: str, patterns: List[re.Pattern]) -> Optional[str]:
    for pat in patterns:
        m = pat.search(text)
        if m:
            # return last group if available, else whole match
            if m.groups():
                # pick first non-empty group
                for g in m.groups():
                    if g and g.strip():
                        return normalize_spaces(g)
                return normalize_spaces(m.group(0))
            return normalize_spaces(m.group(0))
    return None


def parse_sticker_text(vin: str, text: str) -> StickerData:
    t = text
    data = StickerData(VIN=vin, StickerURL=COLLEGE_PARK_STICKER_URL.format(vin=vin))

    # VIN
    vin_found = extract_with_regex(t, [
        re.compile(r"\bVIN[:\s]+([A-HJ-NPR-Z0-9]{17})", re.IGNORECASE),
        re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b"),
    ])
    if vin_found and len(vin_found) == 17:
        data.VIN = vin_found

    # Trim (Calligraphy, Limited, SEL, XRT, etc.)
    data.Trim = extract_with_regex(t, [
        re.compile(r"\bTrim[:\s]*(Calligraphy|Limited|SEL(?: Convenience)?|XRT(?: Pro)?)\b", re.IGNORECASE),
        re.compile(r"\b(?:Hyundai\s+)?Palisade[:\s]+(Calligraphy|Limited|SEL(?: Convenience)?|XRT(?: Pro)?)\b", re.IGNORECASE),
        re.compile(r"\bModel[:\s]*Palisade\s+(Calligraphy|Limited|SEL(?: Convenience)?|XRT(?: Pro)?)\b", re.IGNORECASE),
    ])

    # Drivetrain (AWD/FWD)
    data.Drivetrain = extract_with_regex(t, [
        re.compile(r"\bDrivetrain[:\s]*(AWD|FWD|4WD|Front[-\s]?Wheel|All[-\s]?Wheel)\b", re.IGNORECASE),
        re.compile(r"\b(AWD|FWD)\b", re.IGNORECASE),
    ])

    # Seats: look for "7-passenger" or "8-passenger" or "seats 7/8"
    data.Seats = extract_with_regex(t, [
        re.compile(r"\b(7|8)\s*[- ]?\s*passenger\b", re.IGNORECASE),
        re.compile(r"\bseats?\s*(7|8)\b", re.IGNORECASE),
        re.compile(r"\b(7|8)\s*[- ]?\s*seat(er)?\b", re.IGNORECASE),
        re.compile(r"\bCaptain'?s Chairs\b", re.IGNORECASE),  # signal for 7-seat
    ])

    # Engine, Horsepower
    data.Engine = extract_with_regex(t, [
        re.compile(r"\bEngine[:\s]*([0-9.]+L.*?V[- ]?6.*?)\b", re.IGNORECASE),
        re.compile(r"\b(3\.5L\s+V6.*?)\b", re.IGNORECASE),
    ])
    data.Horsepower = extract_with_regex(t, [
        re.compile(r"\b(\d{3})\s*hp\b", re.IGNORECASE),
        re.compile(r"\bhorsepower[:\s]*(\d{3})\b", re.IGNORECASE),
    ])

    # Exterior / Interior colors
    data.ExteriorColor = extract_with_regex(t, [
        re.compile(r"\bExterior(?:\s+Color)?[:\s]*([A-Za-z][A-Za-z\s/-]+)\b", re.IGNORECASE),
        re.compile(r"\bColor[:\s]*Exterior[:\s]*([A-Za-z][A-Za-z\s/-]+)\b", re.IGNORECASE),
    ])
    data.InteriorColor = extract_with_regex(t, [
        re.compile(r"\bInterior(?:\s+Color)?[:\s]*([A-Za-z][A-Za-z\s/-]+)\b", re.IGNORECASE),
        re.compile(r"\bColor[:\s]*Interior[:\s]*([A-Za-z][A-Za-z\s/-]+)\b", re.IGNORECASE),
    ])

    # Prices: Base MSRP, Destination, Total MSRP
    base = extract_with_regex(t, [
        re.compile(r"\bBase MSRP[:\s]*(" + CURRENCY_RE + r")", re.IGNORECASE),
        re.compile(r"\bMSRP[:\s]*(" + CURRENCY_RE + r")", re.IGNORECASE),
        re.compile(r"\bVehicle Price[:\s]*(" + CURRENCY_RE + r")", re.IGNORECASE),
    ])
    dest = extract_with_regex(t, [
        re.compile(r"\bDestination(?: Charge| Fee)?[:\s]*(" + CURRENCY_RE + r")", re.IGNORECASE),
        re.compile(r"\bFreight(?: & Handling)?[:\s]*(" + CURRENCY_RE + r")", re.IGNORECASE),
    ])
    total = extract_with_regex(t, [
        re.compile(r"\bTotal MSRP[:\s]*(" + CURRENCY_RE + r")", re.IGNORECASE),
        re.compile(r"\bTotal Price[:\s]*(" + CURRENCY_RE + r")", re.IGNORECASE),
    ])

    data.BaseMSRP = parse_currency(base)
    data.DestinationCharge = parse_currency(dest)
    data.TotalMSRP = parse_currency(total)

    # Packages and Options: capture blocks under headings
    # Heuristics: search for headings and grab following lines until blank line or new section
    def extract_section_block(label_patterns: List[re.Pattern]) -> Optional[str]:
        lines = t.splitlines()
        for i, line in enumerate(lines):
            for pat in label_patterns:
                if pat.search(line):
                    # collect subsequent non-empty lines that don't look like totals
                    collected = []
                    for j in range(i+1, min(i+25, len(lines))):
                        ln = lines[j].strip()
                        if not ln:
                            break
                        if re.search(r"^Total|^MSRP|^Destination", ln, re.IGNORECASE):
                            break
                        # simple filter to avoid repeating headings
                        if re.search(r":\s*$", ln):
                            continue
                        collected.append(ln)
                    if collected:
                        return "; ".join(normalize_spaces(x) for x in collected)
        return None

    data.Packages = extract_section_block([
        re.compile(r"\b(Package|Packages|Optional Packages)\b", re.IGNORECASE),
        re.compile(r"\bInstalled Packages\b", re.IGNORECASE),
    ])
    data.Options = extract_section_block([
        re.compile(r"\b(Options|Installed Options|Additional Options)\b", re.IGNORECASE),
        re.compile(r"\bAccessories\b", re.IGNORECASE),
    ])

    # Parse notes for debugging
    notes = []
    if not data.Trim:
        notes.append("Trim not detected")
    if not data.Seats:
        notes.append("Seats not detected")
    if not data.Packages and not data.Options:
        notes.append("Packages/Options not detected")
    data.ParseNotes = "; ".join(notes) if notes else None

    return data


def process_vins(vins: List[str], out_csv: str, delay_sec: float = 0.5) -> None:
    fieldnames = [f.name for f in dataclasses.fields(StickerData)]
    # But we constructed StickerData via dataclass; import dataclasses for fields
    import dataclasses

    rows: List[Dict[str, Optional[str]]] = []
    for vin in vins:
        vin = vin.strip()
        if not vin:
            continue
        try:
            pdf_bytes = fetch_sticker_pdf(vin)
            text = pdf_to_text(pdf_bytes)
            if not text or len(text.strip()) < 50:
                # Potentially scanned; leave a note
                sd = StickerData(VIN=vin, StickerURL=COLLEGE_PARK_STICKER_URL.format(vin=vin),
                                 ParseNotes="Sticker text extraction empty; may be scanned image")
            else:
                sd = parse_sticker_text(vin, text)
            rows.append(asdict(sd))
        except Exception as e:
            rows.append(asdict(StickerData(VIN=vin, StickerURL=COLLEGE_PARK_STICKER_URL.format(vin=vin),
                                           ParseNotes=f"Error: {e}")))
        time.sleep(delay_sec)

    # Write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [f.name for f in dataclasses.fields(StickerData)])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {out_csv}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Hyundai window sticker scraper for College Park Hyundai (VIN-driven).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--vin", help="Single VIN to fetch")
    g.add_argument("--vin-file", help="Path to a file with one VIN per line")
    p.add_argument("--out", default="stickers.csv", help="Output CSV path")
    p.add_argument("--delay", type=float, default=0.5, help="Delay between requests to be polite")
    # p.add_argument("--enable-ocr", action="store_true", help="Enable OCR fallback (requires pytesseract + pdf2image)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    vins: List[str] = []
    if args.vin:
        vins = [args.vin]
    else:
        with open(args.vin_file, "r", encoding="utf-8") as f:
            vins = [ln.strip() for ln in f if ln.strip()]

    process_vins(vins, args.out, delay_sec=args.delay)


if __name__ == "__main__":
    main()
