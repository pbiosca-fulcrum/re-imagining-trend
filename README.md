# Re-Imagining Trend

This project implements a strategy that trades US stocks based on trend analysis using CNN models. It covers:

- Data loading and preprocessing (including synthetic data generation if real data is missing)
- Chart generation (OHLC images) for CNN input
- CNN model training experiments
- Portfolio construction and analysis

## Project Structure

```
my_trend_project/
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── analysis_lib.py
│   │   └── regression_tables.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── chart_dataset.py
│   │   ├── chart_library.py
│   │   ├── dgp_config.py
│   │   ├── equity_data.py
│   │   └── generate_chart.py
│   ├── experiments/
│   │   ├── __init__.py
│   │   └── cnn_experiment.py
│   ├── model/
│   │   ├── __init__.py
│   │   └── cnn_model.py
│   ├── portfolio/
│   │   ├── __init__.py
│   │   └── portfolio.py
│   └── utils/
│       ├── __init__.py
│       ├── cache_manager.py
│       └── config.py
└── tests/
    ├── __init__.py
    └── test_some_feature.py
```

## Getting Started

### Prerequisites

- Python 3.8 or later

### Installation

1. Clone the repository and navigate into it:
   ```bash
   git clone https://github.com/pbiosca-fulcrum/re-imagining-trend.git
   cd re-imagining-trend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Unix or MacOS
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

Run the main entry point:
```bash
python -m src.main
```
This will execute the full pipeline:
- Generate chart data for US stocks (or use synthetic data if needed)
- Train CNN models with specified configurations
- (Optionally) construct and evaluate a trading portfolio based on model predictions

## Testing

Unit tests are located in the `tests/` directory. To run them with:
```bash
pytest
```
or
```bash
python -m unittest discover
```

## Project Notes

- **Data**: Raw data should be placed in the appropriate directories. If no data is found, synthetic data is generated automatically.
- **Configuration**: Adjust paths and hyperparameters in `src/utils/config.py`.
- **Portfolio**: The portfolio module builds decile portfolios based on model predictions and computes performance metrics such as Sharpe ratios.

<!-- ## License

This project is provided without any license. Feel free to modify or distribute it as needed. -->