import argparse
import pandas as pd 
from ais_data_download import AISDataDownloader

def main():
    parser = argparse.ArgumentParser(description="Download AIS data for a specified date range.")
    parser.add_argument('--start-date', type=str, required=True, help="Start date in YYYY-MM-DD format.")
    parser.add_argument('--end-date', type=str, required=True, help="End date in YYYY-MM-DD format.")
    args = parser.parse_args()


    # Parse dates
    try:
        start_date = pd.to_datetime(args.start_date)
        end_date = pd.to_datetime(args.end_date)
    except ValueError as e:
        print(f"Error parsing dates: {e}")
        return

    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date)

    # Initialize downloader and download data
    downloader = AISDataDownloader()
    downloader.download_date_range(date_range)
    downloader.delete_zip_files()
    print("Download and cleanup complete.")

if __name__ == "__main__":
    main()