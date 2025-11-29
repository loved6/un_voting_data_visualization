# UN Voting Data Visualization

The [UN Voting Data Visualization](https://github.com/loved6/un_voting_data_visualization) project provides a dashboard to visualize voting patterns in the United Nations General Assembly (UNGA) and Security Council (UNSC). The dashboard allows users to explore voting data and compare two different time periods using various visualizations, including scatter plots, treemaps, and statistical tables.

## Project Directory Structure

The project is organized as follows:

```string
un_voting_data_visualization/
├── dataset/ # Automatically created directory and downloaded .csv files
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

### Option 1: Docker (Recommended)

The easiest way to run the application is using Docker:

```bash
# Clone the repo and navigate to directory
git clone https://github.com/loved6/un_voting_data_visualization.git
cd un_voting_data_visualization

# Using the convenience script
./run-docker.sh

# Or using docker-compose directly
docker-compose up --build
```

Then navigate to `http://localhost:8050/` to view the dashboard.

For detailed Docker instructions, see [DOCKER.md](DOCKER.md).

### Option 2: Local Installation

To set up the environment locally, run the following commands from this directory:

```bash
git clone https://github.com/loved6/un_voting_data_visualization.git
cd un_voting_data_visualization/src
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Dashboard

### With Docker

```bash
./run-docker.sh
# Or: docker-compose up
```

### Locally

To run the dashboard locally, execute:

```bash
python app.py
```

In a browser, navigate to `http://localhost:8050/` (Docker) or `http://127.0.0.1:8050/` (local) to view the dashboard.

## Example: BRICS and G7 International Organizations

To visualize the BRICS and G7 international organizations, use the following string in the country groups highlight text field:

```string
BRA, RUS, IND, CHN, ZAF; CAN, FRA, DEU, ITA, JPN, GBR, USA
```

## Manual Download of Latest Datasets

Automatic downloading of datasets is built in. However, you may download the latest `.csv` datasets from the following links:

- [UN General Assembly Voting Data](http://digitallibrary.un.org/record/4060887)
- [UN Security Council Voting Data](http://digitallibrary.un.org/record/4055387)

Once downloaded, place the `.csv` files in the `dataset` directory within this project.
