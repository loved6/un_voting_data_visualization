# UN Voting Data Visualization

The [UN Voting Data Visualization project](https://github.com/loved6/un_voting_data_visualization) provides a dashboard to visualize voting patterns in the United Nations General Assembly (GA) and Security Council (SC). The dashboard allows users to explore voting data and compare two different time periods using various visualizations, including scatter plots, treemaps, and statistical tables.

## Project Directory Structure

The project is organized as follows:

```string
love_final_project/
├── dataset/ # Create and place downloaded .csv files here
│   ├── yyyy_mm_dd_ga_voting.csv
│   └── yyyy_mm_dd_sc_voting.csv
├── report/
│   └── UN_Voting_Data_Visualization.pdf
├── src/
│   ├── app.py
│   ├── load_data.py
│   └── requirements.txt
└── README.md
```

## Report

A detailed report explaining the methodology, case study, and results can be found in the `report` directory as `UN_Voting_Data_Visualization.pdf`.

## Installation

To set up the environment, run the following commands from this directory:

```bash
cd src
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Download Latest Datasets

Download the latest `.csv` datasets from the following links:

- [UN General Assembly Voting Data](http://digitallibrary.un.org/record/4060887)
- [UN Security Council Voting Data](http://digitallibrary.un.org/record/4055387)

Once downloaded, place the `.csv` files in the `dataset` directory within this project.

## Running the Dashboard

To run the dashboard, execute:

```bash
python app.py
```

In a browser, navigate to `http://127.0.0.1:8050/` to view the dashboard.

## BRICS and G7 International Organizations

To visualize the BRICS and G7 international organizations, use the following string in the country groups highlight text field:

```string
BRA, RUS, IND, CHN, ZAF; CAN, FRA, DEU, ITA, JPN, GBR, USA
```
