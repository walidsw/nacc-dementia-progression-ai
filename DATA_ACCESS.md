# NACC Data Access Instructions

## Why is the data not included?

This research relies on the Uniform Data Set (UDS) provided by the National Alzheimer's Coordinating Center (NACC). The NACC dataset contains sensitive, longitudinal clinical records from human subjects across multiple Alzheimer's Disease Research Centers. Due to strict patient privacy regulations and the formal Data Use Agreement (DUA) required by NACC, researchers are legally prohibited from redistributing or publicly sharing the raw data files.

## How to Apply for NACC Data Access

To reproduce the results in this repository, you must independently apply for access to the NACC dataset. Access is typically granted to qualified researchers at academic or research institutions.

1. Go to the NACC Data Request Portal: https://naccdata.org/
2. Create an account using your institutional email address.
3. Submit a Data Request for the NACC Uniform Data Set (UDS). You will need to provide a brief summary of your intended research.
4. You must also request access to the ADSP-PHC (Alzheimer's Disease Sequencing Project - Phenotype Harmonization Consortium) supplementary data.
5. Once your request is approved, you will be asked to sign the NACC Data Use Agreement (DUA).
6. After signing the DUA, you will be granted access to download the requested files.

## Required Files for Reproducibility

Once you have access, download the following specific files and folders from the NACC portal:

1. investigator_nacc71.csv (The main UDS investigator dataset)
2. The entire ADSP-PHC-122024-investigator directory and all of its subfolders.

## Setting Up Your Local Directory

For the preprocessing script (data_create.py) to work without modification, place the downloaded files into a folder named Data/ at the root of this repository exactly like this:

dementia-progression-forecasting/
├── Data/
│   ├── investigator_nacc71.csv
│   └── ADSP-PHC-122024-investigator/
│       ├── Biomarker/
│       ├── Cognition/
│       ├── Imaging_DTI/
│       ├── Imaging_FLAIR/
│       ├── Imaging_PET/
│       ├── Imaging_T1/
│       ├── Neuropath/
│       └── Vascular_Risk/
├── data_create.py
├── train_model.py
└── ...

Once this structure is in place, run python data_create.py successfully.

## Legal Disclaimer

All use of the NACC data must comply with the terms and conditions of the NACC Data Use Agreement. The authors of this repository take no responsibility for the improper use, handling, or redistribution of NACC data by third parties. It is solely your responsibility to ensure that your data storage, analysis, and publication practices adhere to the NACC DUA and your local Institutional Review Board (IRB) guidelines.
