import time
import csv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

def convert_market_cap(value_str):
    """Convert abbreviated market cap (e.g. '34.92B', '251.25M', '887.88K') to a float."""
    value_str = value_str.strip().replace(",", "")
    multiplier = 1
    if value_str.endswith('B'):
        multiplier = 1e9
        value_str = value_str[:-1]
    elif value_str.endswith('M'):
        multiplier = 1e6
        value_str = value_str[:-1]
    elif value_str.endswith('K'):
        multiplier = 1e3
        value_str = value_str[:-1]
    try:
        return float(value_str) * multiplier
    except Exception as e:
        print(f"Conversion error for '{value_str}':", e)
        return None

def scrape_current_page(driver):
    """Scrape the ticker and market cap from the table on the current page."""
    page_data = []
    try:
        table = driver.find_element(By.ID, "main-table")
        rows = table.find_elements(By.TAG_NAME, "tr")
        for row in rows[1:]:
            cols = row.find_elements(By.TAG_NAME, "td")
            if len(cols) >= 4:
                try:
                    ticker = cols[0].find_element(By.TAG_NAME, "a").text.strip()
                except:
                    ticker = cols[0].text.strip()
                market_cap_str = cols[3].text.strip()
                market_cap_value = convert_market_cap(market_cap_str)
                page_data.append({"Ticker": ticker, "MarketCap": market_cap_value})
        return page_data
    except Exception as e:
        print("Error scraping current page:", e)
        return page_data

def main():
    url = "https://stockanalysis.com/stocks/"
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(5)

    all_data = []
    page = 1
    while True:
        print(f"Scraping page {page} …")
        page_data = scrape_current_page(driver)
        if not page_data:
            print("No data found on page, exiting loop.")
            break
        all_data.extend(page_data)
        try:
            try:
                overlay = driver.find_element(By.XPATH, "//div[contains(@class, 'fixed') and contains(@class, 'bg-gray-500/50')]")
                driver.execute_script("arguments[0].remove();", overlay)
                print("Overlay removed.")
                time.sleep(1)
            except Exception as overlay_err:
                pass
            next_button = driver.find_element(By.XPATH, "//button[span[contains(text(),'Next')]]")
            if next_button.get_attribute("disabled") is not None:
                print("Next button is disabled. Reached last page.")
                break
            next_button.click()
            page += 1
            time.sleep(5)
        except Exception as e:
            print("Could not find or click the Next button. Exiting loop.", e)
            break
    driver.quit()

    csv_filename = "tickers_marketcap.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        fieldnames = ["Ticker", "MarketCap"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)
    print(f"Scraping complete. {len(all_data)} rows written to {csv_filename}")

if __name__ == "__main__":
    main()
