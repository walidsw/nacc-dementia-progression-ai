# Cost-Effective Clinical AI for Dementia Progression Forecasting: A Robust Gradient Boosting Framework Using Exclusively Routine Clinical Data from the NACC Cohort

**Author:** MD Walid Waccub Swadhin
**Affiliation:** Department of Computer Science and Engineering, Rajshahi University of Engineering and Technology (RUET), Rajshahi, Bangladesh
**Journal:** Computers in Biology and Medicine [Elsevier, Q1]
**Status:** Manuscript submitted for review.
**Corresponding Author:** MD Walid Waccub Swadhin

---

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Journal](https://img.shields.io/badge/Journal-Computers%20in%20Biology%20and%20Medicine-orange?style=flat-square)

---

## Abstract

This repository contains the complete codebase and reproducibility materials for our framework that predicts dementia progression using exclusively routine clinical data, without relying on expensive or invasive neuroimaging (MRI/PET) or cerebrospinal fluid (CSF) biomarkers. The final evaluated configuration is a base XGBoost model without SMOTE; BorderlineSMOTE and a weighted XGBoost/LightGBM/CatBoost ensemble were evaluated as ablations. Validated using a rigorous patient-level GroupShuffleSplit to prevent data leakage, the final model achieves 88.59% accuracy and an 85.57% macro F1-score on an independent test set. The repository includes comprehensive fairness audits and SHAP-based interpretability analysis for clinical transparency.

---

## Key Results

Our framework demonstrated robust discriminative performance across all diagnostic stages, with AUC values ranging from 0.952 to 0.989. Error severity analysis found that 94.0% of misclassifications were only "1-stage off."

| Metric / Diagnostic Class | F1-Score / Value |
| :--- | :--- |
| **Normal (CDR 0)** | 0.94 |
| **MCI (CDR 0.5)** | 0.84 |
| **Mild Dementia (CDR 1.0)** | 0.77 |
| **Severe Dementia (CDR ≥2.0)** | 0.88 |
| **Overall Accuracy** | **88.59%** |
| **Macro F1-Score** | **85.57%** |

---

## Repository Structure

```
.
├── dementia_progression_paper.tex        # LaTeX source of the manuscript
├── dementia_progression_paper.pdf        # Compiled manuscript in PDF format
├── train_model.py                        # Main script for model training and evaluation
├── data_create.py                        # Script for data preprocessing and feature engineering
├── clinical-data-single-validation.ipynb # Full Jupyter notebook with validation and figures
├── requirements.txt                      # Python dependencies required to run the code
├── README.md                             # This file
├── LICENSE                               # MIT License for the codebase
├── DATA_ACCESS.md                        # Instructions for obtaining the required NACC dataset
├── elsarticle.cls                        # LaTeX document class for Elsevier journals
├── multirow.sty                          # LaTeX style file required for table formatting
├── output/                               # Directory containing all generated result figures and CSVs
│   ├── roc_curves.png                    # ROC curve analysis figure
│   ├── confusion_matrix.png              # Confusion matrix figure
│   └── ...                               # Additional figures and metrics
├── Data/                                 # [NOT INCLUDED DUE TO DUA] Directory for raw NACC data
└── Data_Details/                         # [NOT INCLUDED DUE TO DUA] NACC documentation PDFs
```

---

## Installation

To set up the environment and reproduce the results, clone this repository and install the required Python dependencies:

```bash
git clone https://github.com/yourusername/dementia-progression-forecasting.git
cd dementia-progression-forecasting
pip install -r requirements.txt
```

---

## Usage

WARNING: The NACC dataset is not included in this repository due to a strict Data Use Agreement (DUA). You must obtain the data independently and place it in the Data/ folder before running the scripts. See the Data Access section below for details.

Once the data is correctly placed in the Data/ folder, run the pipeline in the following order:

Step 1: Preprocess the data and engineer features:
python data_create.py

Step 2: Train the XGBoost model and generate all evaluations:
python train_model.py

Alternatively, run the complete pipeline interactively using the provided Jupyter notebook:
jupyter notebook clinical-data-single-validation.ipynb

---

## Data Access

The raw dataset used in this study is the National Alzheimer's Coordinating Center (NACC) Uniform Data Set. Due to patient privacy and the NACC Data Use Agreement, we cannot share these files publicly.

For exact step-by-step instructions on how to apply for access at https://naccdata.org/ and correctly format the downloaded files to work with our code, please read DATA_ACCESS.md.

---

## Citation

If you use our code, framework, or methodology in your research, please cite our paper:

@article{swadhin2026costeffective,
  title={Cost-Effective Clinical AI for Dementia Progression Forecasting: A Robust Gradient Boosting Framework Using Exclusively Routine Clinical Data from the NACC Cohort},
  author={Swadhin, MD Walid Waccub},
  journal={Computers in Biology and Medicine},
  year={2026},
  publisher={Elsevier},
  note={Manuscript submitted for review. DOI to be assigned upon acceptance.}
}

---

## License

The source code in this repository is licensed under the MIT License.

Data Disclaimer: The data used by this code is separately governed by the NACC Data Use Agreement. The MIT License applies exclusively to the code and does not grant any rights to the NACC dataset.

---

## Contact

For any questions regarding the code, methodology, or paper, please contact the author:

MD Walid Waccub Swadhin: waccub@gmail.com
