from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required

from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.io import curdoc
from bokeh.layouts import row,column
from bokeh.models import TextInput, Button, DatePicker, MultiChoice

import math
import datetime as dt
import numpy as np
import yfinance as yf

import pandas as pd
from datetime import *

import time
import queue
from threading import Thread

from yahoo_fin.stock_info import *


# Create your views here.

# @login_required()
def home(request):
    sp500url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    data_table = pd.read_html(sp500url)[0]
    sp500_symbols = data_table["Symbol"].to_list()
    print(type(sp500_symbols))
    return render(request,'services_home.html', {'sp500_symbols': sp500_symbols})

# @login_required()
def download_dataset(request):
    ticker = request.GET.get('ticker')
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    # print(ticker)
    
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Convert data to CSV
    csv_data = data.to_csv(index=False)
    
    # Create the HttpResponse object with CSV data
    response = HttpResponse(csv_data, content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{ticker}_dataset.csv"'
    
    return response

# @login_required()
def polyfit_ma_predictor(request):
    ticker = request.GET.get('ticker')
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')

    import yfinance as yf
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    import pandas as pd
    
    from bokeh.plotting import figure
    from bokeh.embed import components
    from bokeh.models import DatetimeTickFormatter

    from datetime import datetime, timedelta

    if not start_date:
        start_date = '2001-01-01'
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # Download the historical stock prices for the ticker
    df = yf.download(ticker, start=start_date, end=end_date)

    # Define x and y variables
    x = np.arange(len(df))
    y = df['Close']

    # Split the data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    # Fit polynomial models with degrees from 1 to 10 and evaluate on the validation set
    degrees = range(1, 11)
    mse_val = []
    for deg in degrees:
        poly_fit = np.polyfit(x_train, y_train, deg)
        poly = np.poly1d(poly_fit)
        y_val_pred = poly(x_val)
        mse = mean_squared_error(y_val, y_val_pred)
        mse_val.append(mse)

    # Select the best model based on the validation set performance
    best_deg = degrees[np.argmin(mse_val)]
    poly_fit = np.polyfit(x, y, best_deg)
    poly = np.poly1d(poly_fit)

    # Plot the curve and the original data using Bokeh
    p = figure(title=f'{ticker} Polynomial Fit (deg={best_deg})', x_axis_label='Date', y_axis_label='Price')
    p.line(df.index, y, legend_label=ticker, line_width=2)
    p.line(df.index, poly(x), legend_label=f'Polynomial Fit (deg={best_deg})', line_width=2, line_color='red')

    # Set the x-axis tick formatter to display the date in a desired format
    p.xaxis.formatter = DatetimeTickFormatter(days=["%m/%d/%Y"], months=["%m/%Y"], years=["%Y"])

    # Predict future prices using the polynomial fit
    future_x = pd.date_range(df.index[-1], periods=30, freq='D')
    future_y = poly(np.arange(len(df), len(df)+30))
    p.line(future_x, future_y, legend_label='Future Prices', line_width=2, line_color='green')

    # Generate the JavaScript and HTML components for embedding
    script, div = components(p)

    # Pass the script and div to the HTML template
    context = {
        'script': script,
        'div': div,
    }
    return render(request, 'polycurve_fit.html', context)

# @login_required()
def bayesian_predictor(request):
    ticker = request.GET.get('ticker')
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')

    import yfinance as yf
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import BayesianRidge
    from sklearn.model_selection import train_test_split
    from bokeh.plotting import figure
    from bokeh.embed import components
    from bokeh.models import DatetimeTickFormatter

    from datetime import datetime, timedelta

    if not start_date:
        start_date = '2001-01-01'
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # Download the historical stock prices for the ticker
    df = yf.download(ticker, start=start_date, end=end_date)

    # Define x and y variables
    x = np.arange(len(df))
    y = df['Close']

    # Split the data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

    # Fit Bayesian linear regression model
    bayesian_reg = BayesianRidge()
    bayesian_reg.fit(x_train.reshape(-1, 1), y_train)

    # Predict future prices using the Bayesian regression model
    future_x = pd.date_range(df.index[-1], periods=30, freq='D')
    future_x_int = np.arange(len(df), len(df)+30).reshape(-1, 1)
    future_y, future_std = bayesian_reg.predict(future_x_int, return_std=True)

    # Plot the curve and the original data using Bokeh
    p = figure(title=f'{ticker} Bayesian Regression', x_axis_label='Date', y_axis_label='Price')
    p.line(df.index, y, legend_label=ticker, line_width=2)
    p.line(future_x, future_y, legend_label='Future Prices', line_width=2, line_color='green')
    p.patch(np.concatenate([future_x, future_x[::-1]]), np.concatenate([future_y - future_std, (future_y + future_std)[::-1]]), color='gray', alpha=0.2)

    # Set the x-axis tick formatter to display the date in a desired format
    p.xaxis.formatter = DatetimeTickFormatter(days=["%m/%d/%Y"], months=["%m/%Y"], years=["%Y"])

    # Generate the JavaScript and HTML components for embedding
    script, div = components(p)

    # Pass the script and div to the HTML template
    context = {
        'script': script,
        'div': div,
    }
    return render(request, 'bayesianRegression.html', context)

def sentiment_ticker(request):
    ticker = request.GET.get('ticker')
    import yfinance as yf
    import pandas as pd
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from newsapi import NewsApiClient

    # Initialize NLTK
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()

    # Initialize News API client
    newsapi = NewsApiClient(api_key='51309dd5599e4c349ff05b29999110c3')

    # Define the stock ticker
    # ticker = 'AAPL'

    ticker_obj = yf.Ticker(ticker)
    company_name = ticker_obj.info['longName'] 

    # Download historical data for the stock
    # data = yf.download(ticker, period='max')

    # Get news articles related to the stock ticker
    news = newsapi.get_everything(q=company_name, language='en', sort_by='relevancy')

    # Preprocess the news text and compute sentiment scores
    sentiments = []
    articles = []
    for article in news['articles']:
        # print(article['title'])
        title = article['title']
        description = article['description']
        if description is not None:
            text = title + ' ' + description
        else:
            text = title
        text = text.lower().replace('\n', ' ')
        sentiment = sia.polarity_scores(text)['compound']
        sentiments.append(sentiment)
        # articles.append((article['title'], article['url'], article['description']))
        articles.append(article)

    # Compute the average sentiment score for the news data
    avg_sentiment = sum(sentiments) / len(sentiments)

    # Predict the stock trend based on the sentiment score
    if avg_sentiment >= 0.05:
        trend = 'Up'
    elif avg_sentiment <= -0.05:
        trend = 'Down'
    else:
        trend = 'Neutral'

    # Print the predicted trend
    # print('Predicted trend for {}: {}'.format(ticker, trend))
    context = {
        'trend': trend,
        'ticker': ticker,
        'companyName': company_name,
        'articles': articles,
    }
    return render(request, 'tickerSentiment.html', context)

def volatilityShift(request):
    ticker = request.GET.get('ticker')
    import yfinance as yf
    from adtk.data import validate_series
    from adtk.visualization import plot
    from adtk.detector import VolatilityShiftAD
    from bokeh.plotting import figure, show
    from bokeh.models import DatetimeTickFormatter
    from bokeh.embed import components

    """
    VolatilityShiftAD detects shift of volatility level by tracking the difference between standard deviations at two sliding time windows next to each other. Internally, it is implemented as a pipenet with transformer DoubleRollingAggregate.
    """
    data = yf.download(ticker)['Close']

    data = validate_series(data)
    volatility_detector = VolatilityShiftAD(c=6.0, side="positive", window=30)
    anomalies = volatility_detector.fit_detect(data)

    # Create a Bokeh figure
    p = figure(title=f"{ticker} Close Price", x_axis_label='Date', y_axis_label='Price')

    # Format the x-axis tick labels
    p.xaxis.formatter=DatetimeTickFormatter(days=["%Y/%m/%d"], months=["%Y/%m/%d"], years=["%Y/%m/%d"])

    # Plot the data as a line
    p.line(data.index, data.values, line_width=2)

    # Add red circles at the anomaly points
    anomaly_dates = [date for date, anomaly in anomalies.items() if anomaly]
    anomaly_values = [data[date] for date in anomaly_dates]
    p.circle(anomaly_dates, anomaly_values, size=8, color='red')

    # Generate the JavaScript and HTML components for embedding
    script, div = components(p)
    
    # Pass the script and div to the HTML template
    context = {
        "script": script,
        'div': div,
    }
    return render(request, 'volatilityShift.html', context)

def seeCandleStick(request):
    ticker = request.GET.get('ticker')
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    import yfinance as yf
    from bokeh.plotting import figure, show
    from bokeh.layouts import gridplot
    import pandas as pd

    # retrieve stock data
    data = yf.download(ticker, start=start_date, end=end_date)

    # ask user for window values
    windows = request.GET.getlist('windows')
    windows = [int(window.strip()) for window in windows]

    # create Bokeh figure
    p = figure(x_axis_type="datetime", title=f"{ticker} Candlestick Chart")

    # calculate candlestick properties
    inc = data.Close > data.Open
    dec = data.Open > data.Close
    w = 12*60*60*1000 # width of each candlestick (12 hours in milliseconds)
    midpoint = (data.Open + data.Close) / 2
    height = abs(data.Close - data.Open)

    # add candlestick glyphs
    p.segment(data.index, data.High, data.index, data.Low, color="black")
    p.vbar(data.index[inc], w, data.Open[inc], data.Close[inc], fill_color="#00FF00", line_color="black")
    p.vbar(data.index[dec], w, data.Open[dec], data.Close[dec], fill_color="#FF0000", line_color="black")

    # add moving average lines
    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]
    for i, window in enumerate(windows):
        # calculate moving average
        ma = data.Close.rolling(window=window).mean()

        # add line glyph
        p.line(data.index, ma, color=colors[i], legend_label=f"MA{window}")

    # configure legend
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    # Generate the JavaScript and HTML components for embedding
    script, div = components(p)
    
    # Pass the script and div to the HTML template
    context = {
        "script": script,
        'div': div,
    }
    return render(request, 'candleStick.html', context)

def stockPicker(request):
    stock_picker = tickers_sp500()
    return render(request, 'stockPicker.html', {'sp500_symbols': stock_picker})

def stockTracker(request):
    stockPicker = request.GET.getlist('stockPicker')
    print(stockPicker)

    data = {}
    available_stocks = tickers_sp500()
    for i in stockPicker:
        if i in available_stocks:
            pass
        else:
            return HttpResponse("Error")

    n_threads = len(stockPicker)
    thread_list = []
    que = queue.Queue()

    start = time.time()
    # print(start)
    # for i in stockPicker:
    #     result = get_quote_table(i)
    #     data.update({i:result})

    for i in range(n_threads):
        thread = Thread(target= lambda q, arg1: q.put({stockPicker[i]: get_quote_table(arg1)}), args=(que, stockPicker[i]))
        thread_list.append(thread)
        thread_list[i].start()

    for thread in thread_list:
        thread.join()

    while not que.empty():
        result = que.get()
        data.update(result)

    end = time.time()
    time_taken = end-start
    # print(time_taken)
    # print(data)
    return render(request, 'stockTracker.html',{'data':data,'room_name':'track'})

def SMA(request):
    # importing libraries and modules
    import pandas as pd
    import numpy as np
    import datetime as dt
    import yfinance as yf
    from bokeh.models import ColumnDataSource
    from bokeh.plotting import figure, show
    from bokeh.layouts import column

    # setting the ticker symbol and the date range
    # ticker = "TSLA"
    ticker = request.GET.get('ticker')
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    start_date = dt.datetime(2022, 1, 1)
    end_date = dt.date.today()

    # retrieving the data using Yahoo Finance API
    data = yf.download(ticker, start=start_date, end=end_date)

    # computing the slow and fast moving averages
    data["SMA_5"] = data["Close"].rolling(window=5).mean()
    data["SMA_15"] = data["Close"].rolling(window=15).mean()
    data["SMA_ratio"] = data["SMA_15"] / data["SMA_5"]

    # creating a ColumnDataSource object
    source = ColumnDataSource(data)

    # creating the first plot
    p1 = figure(title=f"{ticker} Stock Price, Slow and Fast Moving Average", x_axis_type="datetime", width=800, height=500)
    p1.background_fill_color = 'ghostwhite'

    p1.line(x='Date', y='Close', source=source, legend_label='Close')
    p1.line(x='Date', y='SMA_5', source=source, legend_label='SMA_5', color='red')
    p1.line(x='Date', y='SMA_15', source=source, legend_label='SMA_15', color='green')

    p1.legend.location = "top_left"
    p1.legend.click_policy="hide"

    # creating the second plot
    p2 = figure(title="SMA Ratio", x_axis_type="datetime", width=800, height=200)
    p2.background_fill_color = 'silver'
    p2.line(x='Date', y='SMA_ratio', source=source, legend_label='SMA_Ratio', color='blue')

    p2.legend.location = "top_left"
    p2.legend.click_policy="hide"

    # arranging the plots in a column layout
    layout = column(p1, p2)

    script, div = components(layout)

    context = {
        "script": script,
        "div": div,
    }

    return render(request, 'showSMA.html', context)

def bestDay(request):
    import yfinance as yf
    import requests
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import plotly.express as px
    import seaborn as sns
    from bokeh.embed import components
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource
    sns.set()
    from yahoo_fin import stock_info as si

    sp500url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    data_table = pd.read_html(sp500url)[0]

    sp500_symbols = data_table["Symbol"].to_list()

    end_date = date.today()
    start_date = end_date - pd.DateOffset(365*5) # 5 year data

    stocks_list = si.tickers_sp500() # all s&p500 stocks
    df_sp500 = yf.download(tickers = stocks_list, start = start_date, end = end_date)
    sp500_symbols = df_sp500["Close"].columns.to_list()
    df = df_sp500.sort_index()

    gap_returns = np.log(df["Open"]/df["Close"].shift(1))
    intraday_returns = np.log(df["Close"]/df["Open"])
    df_variation =  df["Adj Close"].pct_change()
    df_volatility=df_variation.rolling(250).std()*100*np.sqrt(250)

    weekday = gap_returns.index.map(lambda x: x.weekday())

    best_day=pd.concat([
        gap_returns.groupby(weekday).mean().T.mean().rename("Gap_return mean"),
        gap_returns.groupby(weekday).std().T.mean().rename("Gap_return std"),
        
        intraday_returns.groupby(weekday).mean().T.mean().rename("IntraDay_return mean"),
        intraday_returns.groupby(weekday).std().T.mean().rename("IntraDay_return std"),
        
        df_volatility.groupby(weekday).mean().T.mean().rename("Volatility"),
    ],axis=1)

    best_day.reset_index(inplace=True)
    best_day["Date"] = best_day["Date"].map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri"})
    best_day.rename(columns={"Date":"Day"},inplace=True)

    fig, axs = plt.subplots(1,2, figsize=(20,5))
    sns.barplot(x=best_day["Day"],y=best_day["Gap_return mean"],ax=axs[0])
    axs[0].set_title("Mean Gap Return per Day of the Week")

    sns.barplot(x=best_day["Day"],y=best_day["IntraDay_return mean"],ax=axs[1])
    axs[1].set_title("Mean IntraDay Return per Day of the Week")

        # convert the Seaborn plot to a Bokeh plot
    p = figure(x_range=best_day["Day"], height=400, width=800, title="Mean Returns per Day of the Week")
    source = ColumnDataSource(best_day)
    p.vbar(x='Day', top='Gap_return mean', source=source, width=0.5, color='blue', legend_label="Gap Returns")
    p.vbar(x='Day', top='IntraDay_return mean', source=source, width=0.5, color='red', legend_label="Intraday Returns")
    p.xaxis.axis_label = "Day of the Week"
    p.yaxis.axis_label = "Mean Return"

    # generate the JavaScript and HTML code needed to embed the Bokeh plot into an HTML template
    script, div = components(p)

    # render the HTML template and pass the Bokeh plot to the context
    context = {'script': script, 'div': div}
    return render(request, 'bestDay.html', context)

def top10Returns(request):
    import yfinance as yf
    import requests
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import plotly.express as px
    import seaborn as sns
    from bokeh.embed import components
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource
    sns.set()
    from yahoo_fin import stock_info as si

    sp500url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    data_table = pd.read_html(sp500url)[0]

    sp500_symbols = data_table["Symbol"].to_list()

    end_date = date.today()
    start_date = end_date - pd.DateOffset(365*5) # 5 year data

    stocks_list = si.tickers_sp500() # all s&p500 stocks
    df_sp500 = yf.download(tickers = stocks_list, start = start_date, end = end_date)
    sp500_symbols = df_sp500["Close"].columns.to_list()
    df = df_sp500.sort_index()

    gap_returns = np.log(df["Open"]/df["Close"].shift(1))
    intraday_returns = np.log(df["Close"]/df["Open"])
    df_variation =  df["Adj Close"].pct_change()
    df_volatility=df_variation.rolling(250).std()*100*np.sqrt(250)

    weekday = gap_returns.index.map(lambda x: x.weekday())

    best_day=pd.concat([
        gap_returns.groupby(weekday).mean().T.mean().rename("Gap_return mean"),
        gap_returns.groupby(weekday).std().T.mean().rename("Gap_return std"),
        
        intraday_returns.groupby(weekday).mean().T.mean().rename("IntraDay_return mean"),
        intraday_returns.groupby(weekday).std().T.mean().rename("IntraDay_return std"),
        
        df_volatility.groupby(weekday).mean().T.mean().rename("Volatility"),
    ],axis=1)

    best_day.reset_index(inplace=True)
    best_day["Date"] = best_day["Date"].map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri"})
    best_day.rename(columns={"Date":"Day"},inplace=True)

    df_perCompany=pd.DataFrame( data_table[['Symbol', 'GICS Sector']])
    df_perCompany.rename(columns={"Symbol":"Ticker"},inplace=True)

    for ticker in sp500_symbols:
        df_adjClose_ticker=df["Adj Close"][ticker].dropna()
        if df_adjClose_ticker.shape[0]==0:
            continue
        # creates a pandas Series object called year_index that maps the index of the df_adjClose_ticker DataFrame to the year of each date in the index.
        year_index = df_adjClose_ticker.index.map(lambda x: x.year)

        first_close, last_close = df_adjClose_ticker.iloc[[0,-1]]
        total_return = (last_close/first_close)-1
        first_year = df_adjClose_ticker.index[0].year
        last_year = df_adjClose_ticker.index[-1].year

        years=last_year-first_year+1
        returnPerYear=[]
        for year in range(first_year,last_year+1):
            first_close_year, last_close_year = df_adjClose_ticker[year_index==year].iloc[[0,-1]]
            year_return= (last_close_year/first_close_year)-1
            returnPerYear.append(year_return)
        mean_return_per_year = np.mean(returnPerYear)
        volatility = np.std(returnPerYear)
        df_perCompany.loc[df_perCompany["Ticker"]==ticker,["years","total_return","mean_return_per_year","volatility"]]=years,total_return,mean_return_per_year,volatility
        
    df_perCompany.dropna(inplace=True)

    Rf = 0.01/255
    df_perCompany["Return_Volatility_Ratio"] = (df_perCompany["mean_return_per_year"]*df_perCompany["total_return"])/((df_perCompany["volatility"]-Rf)*df_perCompany["years"])
    top10_companies=df_perCompany.sort_values(by="Return_Volatility_Ratio",ascending=False)[0:10]

    tickers_list = top10_companies['Ticker'].to_list()
    sector_list = top10_companies['GICS Sector'].to_list()
    Return_Volatility_Ratio = top10_companies['Return_Volatility_Ratio'].to_list()

    fig = px.sunburst(top10_companies, path=['GICS Sector', 'Ticker'], values='Return_Volatility_Ratio',
                  color='total_return')
    
    fig.update_layout(
        width=800,  # specify the desired width
        height=600,  # specify the desired height
    )
    
    import plotly.io as pio

    fig_html = pio.to_html(fig, full_html=False)


    context = {'tickers': tickers_list,
               'sectors': sector_list,
               'returns': Return_Volatility_Ratio,
               'fig': fig_html}

    return render(request, 'top10Returns.html', context)

def top10Sectors(request):
    import yfinance as yf
    import requests
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import plotly.express as px
    import seaborn as sns
    from bokeh.embed import components
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource
    sns.set()
    from yahoo_fin import stock_info as si

    sp500url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    data_table = pd.read_html(sp500url)[0]

    sp500_symbols = data_table["Symbol"].to_list()

    end_date = date.today()
    start_date = end_date - pd.DateOffset(365*5) # 5 year data

    stocks_list = si.tickers_sp500() # all s&p500 stocks
    df_sp500 = yf.download(tickers = stocks_list, start = start_date, end = end_date)
    sp500_symbols = df_sp500["Close"].columns.to_list()
    df = df_sp500.sort_index()

    gap_returns = np.log(df["Open"]/df["Close"].shift(1))
    intraday_returns = np.log(df["Close"]/df["Open"])
    df_variation =  df["Adj Close"].pct_change()
    df_volatility=df_variation.rolling(250).std()*100*np.sqrt(250)

    weekday = gap_returns.index.map(lambda x: x.weekday())

    best_day=pd.concat([
        gap_returns.groupby(weekday).mean().T.mean().rename("Gap_return mean"),
        gap_returns.groupby(weekday).std().T.mean().rename("Gap_return std"),
        
        intraday_returns.groupby(weekday).mean().T.mean().rename("IntraDay_return mean"),
        intraday_returns.groupby(weekday).std().T.mean().rename("IntraDay_return std"),
        
        df_volatility.groupby(weekday).mean().T.mean().rename("Volatility"),
    ],axis=1)

    best_day.reset_index(inplace=True)
    best_day["Date"] = best_day["Date"].map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri"})
    best_day.rename(columns={"Date":"Day"},inplace=True)

    df_perCompany=pd.DataFrame( data_table[['Symbol', 'GICS Sector']])
    df_perCompany.rename(columns={"Symbol":"Ticker"},inplace=True)

    for ticker in sp500_symbols:
        df_adjClose_ticker=df["Adj Close"][ticker].dropna()
        if df_adjClose_ticker.shape[0]==0:
            continue
        year_index = df_adjClose_ticker.index.map(lambda x: x.year)

        first_close, last_close = df_adjClose_ticker.iloc[[0,-1]]
        total_return = (last_close/first_close)-1
        first_year = df_adjClose_ticker.index[0].year
        last_year = df_adjClose_ticker.index[-1].year

        years=last_year-first_year+1
        returnPerYear=[]
        for year in range(first_year,last_year+1):
            first_close_year, last_close_year = df_adjClose_ticker[year_index==year].iloc[[0,-1]]
            year_return= (last_close_year/first_close_year)-1
            returnPerYear.append(year_return)
        mean_return_per_year = np.mean(returnPerYear)
        volatility = np.std(returnPerYear)
        df_perCompany.loc[df_perCompany["Ticker"]==ticker,["years","total_return","mean_return_per_year","volatility"]]=years,total_return,mean_return_per_year,volatility
        
    df_perCompany.dropna(inplace=True)

    Rf = 0.01/255
    df_perCompany["Return_Volatility_Ratio"] = (df_perCompany["mean_return_per_year"]*df_perCompany["total_return"])/((df_perCompany["volatility"]-Rf)*df_perCompany["years"])
    top10_companies=df_perCompany.sort_values(by="Return_Volatility_Ratio",ascending=False)[0:10]
    # tickers_list = top10_companies['Ticker'].to_list()
    # sector_list = top10_companies['GICS Sector'].to_list()
    # Return_Volatility_Ratio = top10_companies['Return_Volatility_Ratio'].to_list()

    df_perSector=df_perCompany.groupby("GICS Sector").mean()
    Rf = 0.01/255
    df_perSector["Return_Volatility_Ratio"] = (df_perSector["mean_return_per_year"]*df_perSector["total_return"])/((df_perSector["volatility"]-Rf)*df_perSector["years"])
    df_perSector.sort_values("Return_Volatility_Ratio",ascending=False,inplace=True)

    min_ratio=df_perCompany["total_return"].min()
    max_ratio=df_perCompany["total_return"].max()
    total_return_scale = (df_perCompany["total_return"]+1-min_ratio)/(max_ratio-min_ratio)

    fig = px.sunburst(df_perCompany, path=['GICS Sector',"Ticker"], values=total_return_scale,
                    color='volatility')
    
    fig.update_layout(
        width=800,  # specify the desired width
        height=600,  # specify the desired height
    )

    from bokeh.io import show
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource, FactorRange

    # Create a ColumnDataSource for the sector data
    source = ColumnDataSource(df_perSector)

    # Create the figure
    p = figure(x_range=FactorRange(factors=df_perSector.index.tolist()), 
            height=500, 
            width=1000,
            title='Return Volatility Ratio by Sector')

    # Add the vertical bars
    p.vbar(x='GICS Sector', 
        top='Return_Volatility_Ratio', 
        width=0.9, 
        source=source, 
        line_color='white')

    # Set the axis labels
    p.xaxis.axis_label = "Sector"
    p.yaxis.axis_label = "Return Volatility Ratio"

    # Show the figure
    # show(p)

    # generate the JavaScript and HTML code needed to embed the Bokeh plot into an HTML template
    script, div = components(p)

    # render the HTML template and pass the Bokeh plot to the context
    context = {'script': script, 'div': div}


    import plotly.io as pio

    fig_html = pio.to_html(fig, full_html=False)


    context = {'fig': fig_html,
               'script': script,
               'div': div}

    return render(request, 'top10Sectors.html', context)

def bestPerfStocks(request):
    import numpy as np
    import yfinance as yf
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from yahoo_fin import stock_info as si
    from bokeh.io import output_notebook, show
    from bokeh.plotting import figure

    sp500url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    data_table = pd.read_html(sp500url)
    data_table[0]

    end_date = date.today()
    start_date = end_date - pd.DateOffset(365*5) # 5 year data

    stocks_list = si.tickers_sp500() # all s&p500 stocks
    Prices_five_year = yf.download(tickers=stocks_list, start=start_date, end=end_date)

    prices_df = Prices_five_year.stack()

    prices_df.to_csv('prices_df1.csv')

    df = df = pd.read_csv('prices_df1.csv', parse_dates=['Date'])

    df.columns = ['Date','Symbol','Adj Close','Close','High','Low','Open','Volume']
    df = df[['Date','Close','Symbol']]

    stocks = df.pivot_table(index='Date', columns='Symbol', values='Close')
    stocks = stocks.dropna(axis=1)

    stocks.index = pd.to_datetime(stocks.index, utc=True)

    stocks = stocks.resample('W').last()

    start = stocks.iloc[0]
    returns = (stocks - start) / start

    best = returns.iloc[-1].sort_values(ascending=False).head()
    worst = returns.iloc[-1].sort_values().head()

    def get_name(symbol):
        name = symbol
        try:
            # Convert the DataFrame to a dictionary
            symbol_to_name = dict(zip(data_table[0]['Symbol'], data_table[0]['Security']))
            # Look up the name based on the symbol
            name = symbol_to_name[symbol]
        except KeyError:
            pass
        return name

    from bokeh.models import ColumnDataSource
    from bokeh.palettes import Category10_10
    from bokeh.transform import factor_cmap
    from bokeh.models import HoverTool

    def plot_stock(symbol, stocks=stocks):
        name = get_name(symbol)
        # Create a Bokeh figure
        p = figure(title=name, x_axis_label='Date', y_axis_label='Price', x_axis_type='datetime')

        # Create a ColumnDataSource from the stocks dataframe for the given symbol
        source = ColumnDataSource(stocks[[symbol]])

        # Plot the line chart
        p.line(x='Date', y=symbol, source=source, legend_label=name, line_width=2, line_color='navy')

        # Add hover tool
        p.add_tools(HoverTool(tooltips=[('Date', '@Date{%F}'), ('Price', '@' + symbol)], formatters={'@Date': 'datetime'}))

        # Show the plot
        # show(p)
        return p


    names1 = pd.DataFrame({'name':[get_name(symbol) for symbol in best.index.to_list()]}, index = best.index)
    best = pd.concat((best, names1), axis=1)

    names2 = pd.DataFrame({'name':[get_name(symbol) for symbol in worst.index.to_list()]}, index = worst.index)
    worst = pd.concat((worst, names2), axis=1)

    best_plot = best_first_symbol = best.index[0]
    worst_plot = worst_first_symbol = worst.index[0]

    # print(best_first_symbol)
    # print(worst_first_symbol)

    best_plot_graph = plot_stock(best_first_symbol, stocks=returns)
    worst_plot_graph = plot_stock(worst_first_symbol, stocks=returns)

     # Embed the Bokeh plots in the HTML template
    best, div1 = components(best_plot_graph)
    worst, div2 = components(worst_plot_graph)

    # Render the HTML template
    return render(request, 'bestPerfStock.html', {'best': best, 'div1': div1, 'worst': worst, 'div2': div2})

def clusterStocks(request):
    import numpy as np
    import yfinance as yf
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from yahoo_fin import stock_info as si
    from bokeh.io import output_notebook, show
    from bokeh.plotting import figure

    sp500url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    data_table = pd.read_html(sp500url)
    data_table[0]

    end_date = date.today()
    start_date = end_date - pd.DateOffset(365*5) # 5 year data

    stocks_list = si.tickers_sp500() # all s&p500 stocks
    Prices_five_year = yf.download(tickers=stocks_list, start=start_date, end=end_date)

    prices_df = Prices_five_year.stack()

    prices_df.to_csv('prices_df1.csv')

    df = df = pd.read_csv('prices_df1.csv', parse_dates=['Date'])

    df.columns = ['Date','Symbol','Adj Close','Close','High','Low','Open','Volume']
    df = df[['Date','Close','Symbol']]

    stocks = df.pivot_table(index='Date', columns='Symbol', values='Close')
    stocks = stocks.dropna(axis=1)

    stocks.index = pd.to_datetime(stocks.index, utc=True)

    stocks = stocks.resample('W').last()

    start = stocks.iloc[0]
    returns = (stocks - start) / start

    best = returns.iloc[-1].sort_values(ascending=False).head()
    worst = returns.iloc[-1].sort_values().head()

    def get_name(symbol):
        name = symbol
        try:
            symbol_to_name = dict(zip(data_table[0]['Symbol'], data_table[0]['Security']))
            name = symbol_to_name[symbol]
        except KeyError:
            pass
        return name

    from bokeh.models import ColumnDataSource
    from bokeh.palettes import Category10_10
    from bokeh.transform import factor_cmap
    from bokeh.models import HoverTool

    def plot_stock(symbol, stocks=stocks):
        name = get_name(symbol)
        # Create a Bokeh figure
        p = figure(title=name, x_axis_label='Date', y_axis_label='Price', x_axis_type='datetime')

        # Create a ColumnDataSource from the stocks dataframe for the given symbol
        source = ColumnDataSource(stocks[[symbol]])

        # Plot the line chart
        p.line(x='Date', y=symbol, source=source, legend_label=name, line_width=2, line_color='navy')

        # Add hover tool
        p.add_tools(HoverTool(tooltips=[('Date', '@Date{%F}'), ('Price', '@' + symbol)], formatters={'@Date': 'datetime'}))

        # Show the plot
        # show(p)
        return p


    # names1 = pd.DataFrame({'name':[get_name(symbol) for symbol in best.index.to_list()]}, index = best.index)
    # best = pd.concat((best, names1), axis=1)

    # names2 = pd.DataFrame({'name':[get_name(symbol) for symbol in worst.index.to_list()]}, index = worst.index)
    # worst = pd.concat((worst, names2), axis=1)

    # best_plot = best_first_symbol = best.index[0]
    # worst_plot = worst_first_symbol = worst.index[0]

    kmeans = KMeans(n_clusters=8, random_state=42)
    kmeans.fit(returns.T)

    clusters = {}
    for l in np.unique(kmeans.labels_):
        clusters[l] = []

    for i,l in enumerate(kmeans.predict(returns.T)):
        clusters[l].append(returns.columns[i])

    # Create a dictionary of the clusters
    cluster_dict = {}
    for c in sorted(clusters):
        cluster_dict[c] = [get_name(symbol)+' ('+symbol+')' for symbol in clusters[c]]

    # Import necessary Bokeh libraries
    from bokeh.layouts import gridplot
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.plotting import figure, show

    # Create a dictionary of the clusters
    cluster_dict = {}
    for c in sorted(clusters):
        cluster_dict[c] = [get_name(symbol)+' ('+symbol+')' for symbol in clusters[c]]

    # Create a list to hold all the plots for the clusters
    all_plots = []

    # Loop through each cluster and create a plot for each stock in the cluster
    for c in sorted(clusters):
        # Create a list to hold all the data sources for each stock in the cluster
        sources = []
        
        # Loop through each stock in the cluster and create a data source for it
        for symbol in clusters[c]:
            name = get_name(symbol)
            source = ColumnDataSource(data=dict(x=returns.index, y=returns[symbol], name=[name]*len(returns.index), symbol=[symbol]*len(returns.index)))
            sources.append(source)
            
        # Create a Bokeh figure
        p = figure(title="Returns (Clusters from PCA components) cluster " + str(c), x_axis_label='Date', y_axis_label='Returns', x_axis_type='datetime')

        # Add a line plot for each stock in the cluster
        for i, source in enumerate(sources):
            p.line(x='x', y='y', source=source, legend_label=cluster_dict[c][i], line_width=2)

        # Add a hover tool
        hover = HoverTool(tooltips=[('Name', '@name'), ('Symbol', '@symbol'), ('Date', '@x{%F}'), ('Returns', '@y{0.2f}%')], formatters={'@x': 'datetime'})
        p.add_tools(hover)

        # Add the plot to the list of all plots
        all_plots.append(p)

    # Create a grid of all the plots and show it
    grid = gridplot(all_plots, ncols=2)
    # show(grid);/

    from django.shortcuts import render
    from bokeh.embed import file_html
    from bokeh.resources import CDN


    html = file_html(grid, CDN, "my plot")
    # return render(request, 'plot.html', {'plot': html})




    # Render the HTML template with the cluster dictionary as a context parameter
    return render(request, 'clusterStock.html', {'clusters': cluster_dict, 'plot': html})

def movAvgPredictor(request):
    stock_ticker = request.GET.get('ticker')
    import os
    import numpy as np
    import pandas as pd
    import yfinance as yf
    import matplotlib.pyplot as plt
    import matplotlib.style as style
    import matplotlib.dates as mdates
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.experimental import enable_hist_gradient_boosting
    from bokeh.plotting import figure, show
    from bokeh.models import ColumnDataSource, HoverTool, NumeralTickFormatter, Legend
    from bokeh.palettes import Category10
    from bokeh.embed import file_html
    from bokeh.resources import CDN

    # Download the stock data for the last year
    data = yf.download(stock_ticker, start='2022-01-23')

    if data.empty:
        print("No data available for the stock ticker symbol: ", stock_ticker)
    else:
        # Convert the date column to a datetime object
        data['Date'] = pd.to_datetime(data.index)

        # Set the date column as the index
        data.set_index('Date', inplace=True)

        # Sort the data by date
        data.sort_index(inplace=True)

        # Get the data for the last year
        last_year = data.iloc[-365:].copy()

        # Calculate the 200-day moving average
        last_year.loc[:,'200MA'] = last_year['Close'].rolling(window=200).mean()

        # Split the data into X (features) and y (target)
        X = last_year[['200MA']]
        y = last_year['Close']

        # Create an HistGradientBoostingRegressor instance
        model = HistGradientBoostingRegressor()

        # Fit the model with the data
        model.fit(X, y)

        # Make predictions for the next 30 days
        future_dates = pd.date_range(start=data.index[-1], periods=30, freq='D')
        future_data = pd.DataFrame(index=future_dates, columns=['200MA'])
        future_data['200MA'] = last_year['200MA'].iloc[-1]
        predictions = model.predict(future_data)
        predictions_df = pd.DataFrame(predictions, index=future_dates, columns=['Close'])

        # Calculate the standard deviation of the last year's close prices
        std_dev = last_year['Close'].std()

        # Generate random values with a standard deviation of 0.5 * the last year's close prices standard deviation
        random_values = np.random.normal(0, 0.2 * std_dev, predictions.shape)

        # Add the random values to the predicted prices
        predictions += random_values
        predictions_df = pd.DataFrame(predictions, index=future_dates, columns=['Close'])

        # Concatenate the last_year and predictions dataframes
        predictions_df = pd.concat([last_year, predictions_df])

        # Calculate 200 day moving average
        predictions_df.loc[:,'MA_200'] = predictions_df['Close'].rolling(window=200).mean()

        # # Set the style to dark theme
        # style.use('dark_background')

        # # Create the plot
        # fig, ax = plt.subplots()

        # # Plot the predicted close prices for the next 30 days
        # ax.plot(predictions_df.index, predictions_df['Close'], color='green' if predictions_df['Close'][-1] >= last_year['Close'][-1] else 'red', label='Predicted')

        # # Plot the actual close prices for the last year
        # ax.plot(last_year.index, last_year['Close'], color='b', label='Actual')

        # ax.plot(predictions_df.index, predictions_df['MA_200'], color='white', label='200 Day MA')

        # # Set x-axis as date format
        # ax.xaxis.set_major_formatter(mdates.DateFormatter("%B %D %Y"))
        # plt.xticks(rotation=45)

        # # Set the x-axis label
        # plt.xlabel('Date')

        # # Set the y-axis label
        # plt.ylabel('Price (USD)')

        # # Set the plot title
        # plt.title(stock_ticker + ' Moving Average Price Prediction')

        # # Show the legend
        # plt.legend()

        # # Show the plot
        # plt.show()

        # Create a Bokeh ColumnDataSource with the data to plot
        from bokeh.plotting import figure, show
        from bokeh.models import DatetimeTickFormatter

        # Create the plot
        fig = figure(title=stock_ticker + ' Moving Average Price Prediction',
                    x_axis_label='Date', y_axis_label='Price (USD)')

        # Plot the predicted close prices for the next 30 days
        fig.line(x=predictions_df.index, y=predictions_df['Close'], line_color='green' if predictions_df['Close'][-1] >= last_year['Close'][-1] else 'red', legend_label='Predicted')

        # Plot the actual close prices for the last year
        fig.line(x=last_year.index, y=last_year['Close'], line_color='blue', legend_label='Actual')

        # Plot the 200 Day MA
        fig.line(x=predictions_df.index, y=predictions_df['MA_200'], line_color='orange', legend_label='200 Day MA')

        # Set x-axis as date format
        fig.xaxis.formatter=DatetimeTickFormatter(months="%B %D %Y")
        fig.xaxis.major_label_orientation = 45

        # Show the legend
        fig.legend.location = "top_left"
        fig.legend.click_policy="hide"

        # Show the plot
        # show(fig)

         # Generate the JavaScript and HTML components for embedding
        script, div = components(fig)

        # Pass the script and div to the HTML template
        context = {
            'script': script,
            'div': div,
        }
        return render(request, 'movAvgPred.html', context)

def movAvgPredictorMACD(request):
    stock_ticker = request.GET.get('ticker')
    import os
    import datetime
    import numpy as np
    import pandas as pd
    import yfinance as yf
    import matplotlib.pyplot as plt
    import matplotlib.style as style
    import matplotlib.dates as mdates
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.experimental import enable_hist_gradient_boosting

    # Ask the user for the stock ticker symbol
    # stock_ticker = input("Enter the stock ticker symbol: ")

    # Get today's date
    today = datetime.datetime.now().date()

    # Subtract 365 days from today's date
    one_year_ago = today - datetime.timedelta(days=365)

    # Use the date one year ago as the start parameter in yf.download()
    data = yf.download(stock_ticker, start=one_year_ago)

    if data.empty:
        print("No data available for the stock ticker symbol: ", stock_ticker)
    else:
        # Convert the date column to a datetime object
        data['Date'] = pd.to_datetime(data.index)

        # Set the date column as the index
        data.set_index('Date', inplace=True)

        # Sort the data by date
        data.sort_index(inplace=True)

        # Get the data for the last year
        last_year = data.iloc[-365:].copy()

        # Calculate the MACD line, signal line, and histogram
        last_year.loc[:,'MACD_Line'] = last_year['Close'].ewm(span=12).mean() - last_year['Close'].ewm(span=26).mean()
        last_year.loc[:,'Signal_Line'] = last_year['MACD_Line'].ewm(span=9).mean()
        last_year.loc[:,'Histogram'] = last_year['MACD_Line'] - last_year['Signal_Line']

        # Split the data into X (features) and y (target)
        X = last_year[['MACD_Line', 'Signal_Line', 'Histogram']]
        y = last_year['Close']

        # Create an HistGradientBoostingRegressor instance
        model = HistGradientBoostingRegressor()

        # Fit the model with the data
        model.fit(X, y)

        # Make predictions for the next 30 days
        future_dates = pd.date_range(start=data.index[-1], periods=30, freq='D')
        future_data = pd.DataFrame(index=future_dates, columns=['MACD_Line','Signal_Line','Histogram'])
        future_data['MACD_Line'] = last_year['MACD_Line'].iloc[-1]
        future_data['Signal_Line'] = last_year['Signal_Line'].iloc[-1]
        future_data['Histogram'] = last_year['Histogram'].iloc[-1]

        predictions = model.predict(future_data)
        predictions_df = pd.DataFrame(predictions, index=future_dates, columns=['Close'])

        # Calculate the standard deviation of the last year's close prices
        std_dev = last_year['Close'].std()

        # Generate random values with a standard deviation of 0.5 * the last year's close prices standard deviation
        random_values = np.random.normal(0, 0.2 * std_dev, predictions.shape)

        # Add the random values to the predicted prices
        predictions += random_values 
        predictions_df = pd.DataFrame(predictions, index=future_dates, columns=['Close'])

        # Concatenate the last_year and predictions dataframes
        predictions_df = pd.concat([last_year, predictions_df])

        # Recalculate MACD line, signal line, and histogram for the next 30 days
        predictions_df.loc[:,'MACD_Line'] = predictions_df['Close'].ewm(span=12).mean() - predictions_df['Close'].ewm(span=26).mean()
        predictions_df.loc[:,'Signal_Line'] = predictions_df['MACD_Line'].ewm(span=9).mean()
        predictions_df.loc[:,'Histogram'] = predictions_df['MACD_Line'] - predictions_df['Signal_Line']

        # Create a new column in the predictions_df DataFrame to store the buy/sell signals, with a default value of "hold"
        predictions_df['Signal'] = 'hold'

        # Iterate through the predictions_df DataFrame and check the values of the MACD_Line and Signal_Line columns
        for i, row in predictions_df.iterrows():
            if i == 0:
                continue
            if row['MACD_Line'] > row['Signal_Line']:
                predictions_df.at[i, 'Direction'] = 'up'
            elif row['MACD_Line'] < row['Signal_Line']:
                predictions_df.at[i, 'Direction'] = 'down'

        # Create the plot
        fig, axs = plt.subplots(2, 1,)

        # Plot the predicted close prices for the next 30 days
        axs[0].plot(predictions_df.index, predictions_df['Close'], color='green' if predictions_df['Close'][-1] >= last_year['Close'][-1] else 'red', label='Predicted')
        axs[0].plot(last_year.index, last_year['Close'], color='blue', label='Actual')
        axs[0].set_title(stock_ticker + " MACD Price Prediction")
        axs[0].set_xticks([])
        axs[0].legend(loc='upper left')

        # Plot the MACD line, signal line, and histogram
        axs[1].plot(predictions_df.index, predictions_df['MACD_Line'], label='MACD Line', color='tab:green')
        axs[1].plot(predictions_df.index, predictions_df['Signal_Line'], label='Signal Line', color='tab:red')
        axs[1].bar(predictions_df.index, predictions_df['Histogram'], label='Histogram', color='tab:blue')
        axs[1].set_title('')
        axs[1].legend(loc='lower left')

        # Create buy and sell signals
        signals = predictions_df[predictions_df['Direction'] != predictions_df['Direction'].shift()].copy()

        # Plot the signal values as scatter points on the second subplot, using a different color for buys and sells
        ax2 = plt.subplot(2, 1, 2)
        ax2.scatter(signals[signals['Direction'] == 'up'].index, signals[signals['Direction'] == 'up']['Signal_Line'], color='green', label='buy')
        ax2.scatter(signals[signals['Direction'] == 'down'].index, signals[signals['Direction'] == 'down']['Signal_Line'], color='red', label='sell')
        
        # Set the x-axis to show dates
        for ax in axs:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)

        # Set the x-axis limits to be the same for both subplots
        axs[0].set_xlim(predictions_df.index[0], predictions_df.index[-1])
        axs[1].set_xlim(predictions_df.index[0], predictions_df.index[-1])

        #Set the y-axis to show labels
        axs[0].set_ylabel('Price (USD)')
        axs[1].set_ylabel('Moving Average Conver/Diver')

        # Show the plot
        # plt.show()
        # fig.savefig('plot.png')
        import mpld3

        html_fig = mpld3.fig_to_html(fig)
        with open('plot.html', 'w') as f:
            f.write(html_fig)

    with open('plot.html', 'r') as f:
        plot_html = f.read()

    context = {'div': plot_html}
    return render(request, 'movAvgPredictorMACD.html', context)

def aggloClustering(request):
    import yfinance as yf
    import pandas as pd
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt
    import seaborn as sns

    import yfinance as yf
    import pandas as pd

    def betas(tickers, stocks, start_date, end_date):
        ticker = yf.download(tickers, start_date, end_date)
        ticker['stock_name'] = tickers
        #daily returns 
        ticker['daily_return'] = ticker['Close'].pct_change(1)
        #standard deviation of the returns
        ticker_std = ticker['daily_return'].std()
        ticker.dropna(inplace=True)
        ticker = ticker[['Close', 'stock_name', 'daily_return']] 

        frames = []
        stds = []
        for i in stocks: 
            data = yf.download(i, start_date, end_date)
            data['stock_name'] = i
            data['daily_return'] = data['Close'].pct_change(1)
            data.dropna(inplace=True)
            data = data[[ 'Close', 'stock_name', 'daily_return']]
            data_std = data['daily_return'].std()
            frames.append(data)
            stds.append(data_std)
        #for each stock calculate its correlation with index
        stock_correlation = []
        for i in frames: 
            correlation = i['daily_return'].corr(ticker['daily_return'])
            stock_correlation.append(correlation)
        #calculating beta 
        betas = []
        for b,i in zip(stock_correlation, stds):
            beta_calc = b * (i/ticker_std)
            betas.append(beta_calc)

        #Dataframe with the results
        dictionary = {stocks[e]: betas[e] for e in range(len(stocks))}
        dataframe = pd.DataFrame([dictionary]).T
        dataframe.reset_index(inplace=True)
        dataframe.rename(
            columns={"index": "Stock_Name", 0: "Beta"},
            inplace=True,)
        return dataframe
    
    companies=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    table  = companies[0]
    df = table[table["Symbol"].str.contains("BRK.B|BF.B") == False]
    ticker_list1 = df['Symbol'].to_list()
    ticker_list1[0:]

    # Call betas
    betas = betas('^GSPC', ticker_list1, '2010-01-01', '2023-05-01')
    
    #assigning Beta column to X
    X = betas[['Beta']]

    #testing number of cluster from 2 to 10 and collecting the silhouette scores
    range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    scores = []
    for n_clusters in range_n_clusters:
        agglom = AgglomerativeClustering(n_clusters=n_clusters)
        agglom.fit(X)
        labels = agglom.labels_
        scores.append(silhouette_score(X, labels))

    # average = sum(scores)/len(scores)

    # Testing Plots to see optimal number of clusters
    # for n_clusters in range_n_clusters:
    #     model = AgglomerativeClustering(n_clusters=n_clusters)
    #     labels = model.fit_predict(X)
    #     # Create scatter plot of data points colored by cluster label
    #     plt.scatter(X, betas['Stock_Name'], c=labels, cmap='rainbow')
    #     plt.xlabel('Beta')
    #     plt.ylabel('Beta')
    #     plt.title(f"n_clusters={n_clusters}")
    #     cluster_counts = np.bincount(labels)
    #     for i in range(n_clusters):
    #         print(f"Cluster {i+1} has {cluster_counts[i]} observations")
    #     plt.yticks([])
    #     plt.show()

    # We choose 4
    # optimal_n_clusters = 4
    # agglom = AgglomerativeClustering(n_clusters=optimal_n_clusters)
    # cluster_labels = agglom.fit_predict(X)
    # betas['Cluster'] = cluster_labels

    # cluster4 = sns.lmplot(data=betas, x='Cluster', y='Beta', hue='Cluster', 
    #                 legend=True, legend_out=True)
    
    # sns.violinplot(x='Cluster', y='Beta', data=betas)


    optimal_n_clusters = 4
    agglom = AgglomerativeClustering(n_clusters=optimal_n_clusters)
    cluster_labels = agglom.fit_predict(X)
    betas['Cluster'] = cluster_labels

    import mpld3
    from matplotlib import pyplot as plt

    cluster4 = sns.lmplot(data=betas, x='Cluster', y='Beta', hue='Cluster', legend=True, legend_out=True)

    # Convert the plot to HTML using mpld3
    html = mpld3.fig_to_html(cluster4.fig)

    # Save the HTML to a file
    with open('myplot.html', 'w') as f:
        f.write(html)

    with open('myplot.html', 'r') as f:
        plot_html = f.read()

     
    # pass the betas dataframe and the plots as context variables to the HTML file
    context = {'betas': betas.to_html(),'div': plot_html}

    # render the HTML file with the context variables
    return render(request, 'aggloClustering.html', context)