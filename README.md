Classification of vessel traffic data (AIS) for the Knowledge-Embedded Learning project between CMU and RTX. Trajectory based ship information provided by the Marine Cadastre at https://hub.marinecadastre.gov/pages/vesseltraffic is used to predict ship classes based solely off the trajectory data. The nominal AIS message format can be viewed at https://www.navcen.uscg.gov/ais-class-a-reports, which are the features that we utilize to classify vessels into classes described at https://coast.noaa.gov/data/marinecadastre/ais/VesselTypeCodes2018.pdf.


# Setup
The code and package requirements were tested with Ubuntu 22.04. 

The requirements in environment.yml can be installed by creating a new conda environment:

```bash
conda env create -f environment.yml
```
## Download trajectory data
A command line interface is provided to download AIS data based on a date range in the following format:

```bash
python cli_ais_downloader.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD
```

The per-day CSVs are stored in /data. 

**Note: The CLI must be run to download the desired date-ranges prior to training.**

#  Repo Structure & Overview
An end-to-end example of how the AIS data is processed, embeddings generated via Gaussian Process Regression, and classificaton carried out is given in 'simple_kernel_classification.ipynb'. For the example shown in the notebook, the CLI should be used to download a date range from 2024-01-01 to 2024-01-07. Some other relevant file and folder descriptions are provided below:

**Kernels for Gaussian Process Regression**

These kernels embed priors into the trajectory regression process under the gpytorch framework https://gpytorch.ai/
- state_space_kernel.py
- multioutput_rbf_linear_gp.py

**Datasets**
- ais_dataloader.py : A pytorch dataset which cleans AIS messages and packages them into a standardized trajectory. The corresponding prediction task is to predict trajectory states based on times for a given ship $t \mapsto x(t)$.
- gp_kernel_ship_classification_dataset.py : A pytorch dataset that takes kernel parameters for each ship and provides the true label. The corresponding prediction task is to predict the true label based on the kernel parameters $\phi^* \mapsto \mathbf{z}$.

**Other**
- /data : contains the downloaded daily AIS csvs (the provided CLI automatically downloads them here)
- /models : Pre-trained Gaussian Process Regression models that are automatically loaded by the notebook if they intersect with the specified date-range.
- /AIS_statistics_utils : various scripts used to analyze statistics on the dataset as a whole
- /logs : tensorboard logs from GP regression & classification tasks

