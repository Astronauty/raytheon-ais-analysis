Classification of vessel traffic data (AIS) for the Knowledge-Embedded Learning project between CMU and RTX.

# Setup
The code and package requirements were tested with Ubuntu 22.04 an M2 Macbook pro on Sequoia 15.11.

The requirements should be installed via conda from requirements.txt

## Download trajectory data
A command line interface is provided to download AIS data based on a date range in the following format:

python cli_ais_downloader.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD

The per-day CSVs are stored in /data.

##  Repo Structure & Overview
/data : contains the downloaded daily AIS csvs
/AIS_statistics_utils : various scripts used to analyze statistics on the dataset as a whole
/logs : tensorboard logs from GP regression & classification tasks

