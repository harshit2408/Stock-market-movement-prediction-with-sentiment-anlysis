# Financial News Analysis and Prediction using Machine Learning

This repository contains a collection of Python scripts and Jupyter notebooks designed to analyze financial news and stock market datasets, and to build predictive models using machine learning algorithms. The project includes creating datasets, sentiment analysis, applying machine learning algorithms, and leveraging deep learning models like LSTM for time-series analysis.

For the financial news collection download the csv files from the following Kaggle link: [Massive Stock News Analysis DB](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests?select=raw_partner_headlines.csv)

To ensure that the tree structure is properly displayed with the correct indentation, place this within a fenced code block like this:

```md
## Project Structure

```bash
.
├── Financial_News
│   ├── analyst_ratings_processed.csv          # Processed financial news dataset
│   ├── raw_analyst_ratings.csv 
│   ├── raw_partner_headlines.csv 
│
├── Datasets
│   ├── sp500_news.csv                         # S&P 500 stock market dataset
│   ├── sp500_sentiment_non_weighted.csv       # Dataset with sentiment scores
│
├── Scripts
│   ├── create_dataset.ipynb                   # Script for creating the initial datasets
│   ├── DF_creation_example.ipynb              # DataFrame creation example
│   ├── Make_dataset.ipynb                     # Dataset creation script
│   ├── LSTM_3.ipynb                           # LSTM model for financial data analysis
│   ├── LSTM_sentim.ipynb                      # LSTM model integrated with sentiment analysis
│   ├── lstm_refined.ipynb                     # Refined version of the LSTM model for prediction of closing price of future dates
│   ├── ML_algos.ipynb                         # Machine learning algorithms for stock prediction
│   ├── sp500_dataset_sentim.ipynb             # Sentiment analysis on S&P 500 dataset
│   ├── utilities.py                           # Utility functions for data processing
│
└── README.md                                  # This file
```

## Dependencies

To run this project, ensure you have the following dependencies installed:

- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- scikit-learn
- keras
- tensorflow
- matplotlib
- seaborn
- nltk

## Process Flow

### 1. Utility Functions

`utilities.py`: This script contains various utility functions used across multiple notebooks for data processing, visualization, and model evaluation.

### 2. Dataset Creation

The initial dataset is created using the following notebooks:

- `create_dataset.ipynb`: This notebook outlines the steps for creating the main dataset from financial news and stock data.
- `Make_dataset.ipynb`: This script creates structured datasets for the machine learning models. It includes data cleaning and preparation steps.
- `DF_creation_example.ipynb`: Provides an example of creating pandas DataFrames from raw data for further analysis.

### 3. Sentiment Analysis

The sentiment analysis is performed on the financial news articles to evaluate their impact on stock prices:

- `sp500_dataset_sentim.ipynb`: Conducts sentiment analysis on the S&P 500 dataset and adds sentiment scores as a feature for model training.

### 4. Modeling with Machine Learning and LSTM

Multiple machine learning and deep learning models are applied to predict stock prices and evaluate the effect of news sentiment:

- `LSTM_3.ipynb`: This notebook contains the initial implementation of an LSTM model for time-series forecasting on financial data.
- `LSTM_sentim.ipynb`: Implements a Long Short-Term Memory (LSTM) model with sentiment analysis integrated into the stock price prediction pipeline.
- `lstm_refined.ipynb`: Refined version of the LSTM model with additional parameters for better prediction accuracy.
- `ML_algos.ipynb`: Contains multiple machine learning algorithms such as Random Forest, Decision Trees, and Support Vector Machines (SVM) for predicting stock market trends.

## Usage

To run the notebooks, follow these steps:

1. Clone the repository:
    ```bash
    [git clone https://github.com/yourusername/financial-news-prediction.git](https://github.com/harshit2408/Stock-market-movement-prediction-with-sentiment-anlysis.git)
    ```

2. Navigate to the folder and open Jupyter Notebook:
    ```bash
    cd financial-news-prediction
    jupyter notebook
    ```

3. Open and run the notebooks in the following order to reproduce the analysis:
    - `Make_dataset.ipynb` (Optional: for sample data creation)
    - `create_dataset.ipynb`
    - `sp500_dataset_sentim.ipynb`
    - `ML_algos.ipynb`
    - `LSTM_3.ipynb`
    - `LSTM_sentim.ipynb`

## Dataset Sources

- **S&P 500 Dataset**: Historical data for S&P 500 companies used for stock price prediction.
- **Financial News Dataset**: [Massive Stock News Analysis DB](https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests?select=raw_partner_headlines.csv)

## Results

The results include:
- Predictive models for stock market trends.
- Insights into the influence of financial news on market movements.
- A robust LSTM model for time-series forecasting with sentiment data as a feature.

## Contributing

If you would like to contribute to this project, feel free to open a pull request. We welcome improvements to the models, data pipelines, and overall structure.

## License

This project is licensed under the MIT License.
