import pandas as pd
import numpy as np
import yfinance
import requests
import lxml
import yfinance as yf
from torch.nn.functional import softmax
from tqdm import tqdm


# Function to download and process financial data using yfinance directly
def financial_dataset(stock, num_of_labels=2, cutoff=0.25,
                      start_date="2010-01-01", end_date="2021-01-01"):
    ''' Downloads financial data for a stock and processes it in the desired format. '''
    
    # Parameter value check
    if (num_of_labels < 2 or num_of_labels > 3):
        return print('Number of labels can be either 2 or 3')
    
    # Download financial data using yfinance
    fin_data = yf.download(stock, start=start_date, end=end_date)
    
    print(f"{stock} financial dataframe dimensions: ", fin_data.shape)
    
    # Initialize price_change column 
    fin_data['Price_change'] = 1
    fin_data['date'] = 0
    dates = fin_data.index
    yesterday = str(dates[0].date())

    # Calculate price changes
    for date in dates[1:]:
        today = str(date.date())

        yesterday_pr = fin_data.loc[yesterday, 'Close']
        today_pr = fin_data.loc[today, 'Close']
        diff = 100 * (today_pr - yesterday_pr) / yesterday_pr

        if (num_of_labels == 3):
            if (diff > cutoff):
                price_change = +1
            elif (diff < -cutoff):
                price_change = -1
            else:
                price_change = 0 
        elif (num_of_labels == 2):
            if (diff > 0): 
                price_change = +1
            elif (diff <= 0):
                price_change = -1 
                                                                                                       
        yesterday = today
        fin_data.loc[today, 'Price_change'] = price_change
        fin_data.loc[today, 'date'] = today

    # Summary of changes
    incr = fin_data[fin_data['Price_change'] == 1].shape[0]
    decr = fin_data[fin_data['Price_change'] == -1].shape[0]
    stable = fin_data[fin_data['Price_change'] == 0].shape[0]
    print(f'Positive changes: {incr}')
    print(f'Negative changes: {decr}')
    print(f'No changes: {stable}')

    # Drop unnecessary columns
    fin_data = fin_data.drop(columns=['Low', 'High', 'Adj Close'], axis=1)
        
    return fin_data

# Function to read financial news from two different datasets
def read_news(stock):
    def read_rph(stock):
        ''' Reads news relevant to 'stock' from the "raw_partner_headlines.csv" file. '''
        csv_path = 'Financial_News/raw_partner_headlines.csv'
        arp = pd.read_csv(csv_path)
        arp = arp.drop(columns=['Unnamed: 0', 'url', 'publisher'], axis=1)
        arp['date'] = arp['date'].apply(lambda x: x.split(' ')[0])
        news = arp[arp['stock'] == stock]
        print(f"The bot found {news.shape[0]} headlines from raw_partner_headlines.csv regarding {stock} stock")
        return news

    def read_arp(stock):
        ''' Reads news relevant to 'stock' from the "analyst_rating_processed.csv" file. '''
        csv_path = 'Financial_News/analyst_ratings_processed.csv'
        arp = pd.read_csv(csv_path)
        arp = arp.drop(columns=['Unnamed: 0'], axis=1)
        arp['date'] = arp['date'].apply(lambda x: str(x).split(' ')[0])
        arp.rename({'title': 'headline'}, axis=1, inplace=True)
        news = arp[arp['stock'] == stock]
        print(f"The bot found {news.shape[0]} headlines from analyst_ratings_processed.csv regarding {stock} stock")
        return news
    
    arp = read_arp(stock)
    rph = read_rph(stock)
    news = pd.concat([rph, arp], ignore_index=True)
    print(f"The bot found {news.shape[0]} headlines in total regarding {stock} stock")
    return news

# Function to merge financial data and news headlines
def merge_fin_news(df_fin, df_news, how='inner'):
    ''' Merges financial data with news data based on the date column '''
    merged_df = df_fin.merge(df_news, on='date', how=how)
    merged_df = merged_df[['date', 'stock', 'Open', 'Close', 'Volume', 'headline', 'Price_change']]
    return merged_df

# Function to analyze sentiment in the news headlines using a pre-trained FinBert model
def sentim_analyzer(df, tokenizer, model):
    ''' Runs sentiment analysis on headlines and adds sentiment columns to the dataframe '''
    for i in tqdm(df.index):
        try:
            headline = df.loc[i, 'headline']
        except:
            return print('\'headline\' column might be missing from dataframe')
        
        input = tokenizer(headline, padding=True, truncation=True, return_tensors='pt')
        output = model(**input)
        predictions = softmax(output.logits, dim=-1)
        
        df.loc[i, 'Positive'] = predictions[0][0].tolist()
        df.loc[i, 'Negative'] = predictions[0][1].tolist()
        df.loc[i, 'Neutral'] = predictions[0][2].tolist()
    
    try:
        df = df[['date', 'stock', 'Open', 'Close', 'Volume', 'headline', 'Positive', 'Negative', 'Neutral', 'Price_change']]
    except:
        pass
    return df

# Function to merge sentiment scores by date
def merge_dates(df):
    ''' Aggregates sentiment scores by date, taking the average for each date '''
    dates_in_df = df['date'].unique()
    
    # Create an empty DataFrame with the required columns
    new_df = pd.DataFrame(columns=['date', 'stock', 'Open', 'Close', 'Volume', 
                                   'Positive', 'Negative', 'Neutral', 'Price_change'])
    
    for date in dates_in_df:
        # Filter data for the current date
        sub_df = df[df['date'] == date]
        
        # Calculate average sentiment scores
        avg_positive = sub_df['Positive'].mean()
        avg_negative = sub_df['Negative'].mean()
        avg_neutral = sub_df['Neutral'].mean()
        
        # Get other values (assuming they are constant per date)
        stock = sub_df.iloc[0]['stock']
        open_price = sub_df.iloc[0]['Open']
        close_price = sub_df.iloc[0]['Close']
        volume = sub_df.iloc[0]['Volume']
        price_change = sub_df.iloc[0]['Price_change']
        
        # Create a new row as a DataFrame
        new_row = pd.DataFrame({
            'date': [date],
            'stock': [stock],
            'Open': [open_price],
            'Close': [close_price],
            'Volume': [volume],
            'Positive': [avg_positive],
            'Negative': [avg_negative],
            'Neutral': [avg_neutral],
            'Price_change': [price_change]
        })
        
        # Concatenate the new row to new_df
        new_df = pd.concat([new_df, new_row], ignore_index=True)
    
    print(f"Dataframe now contains sentiment scores for {new_df.shape[0]} different dates.")
    
    return new_df
