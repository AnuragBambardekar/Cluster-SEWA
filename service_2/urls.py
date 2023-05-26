from django.contrib import admin
from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from django.urls import include
from .views import home, download_dataset
from service_2 import views

urlpatterns = [
    path('', home),
    path('polytrendgraph',views.polyfit_ma_predictor, name='poly_trendgraph'), # trend graph using polynomial curve fitting
    path('bayesianpredictor',views.bayesian_predictor, name='bayesian_predictor'), # trend graph using bayesian ridge learning
    path('tickersentiment',views.sentiment_ticker, name='sentiment_ticker'), # ticker sentiment
    path('volatility',views.volatilityShift, name='volatility'), # volatility shift
    path('candlestick',views.seeCandleStick, name='candlestick'), # candlestick
    path('download', download_dataset, name='download'),
    path('stockPicker', views.stockPicker, name='stockPicker'),
    path('stockTracker', views.stockTracker, name='stockTracker'),
    path('SMA', views.SMA, name='SMA'),
    path('bestDay', views.bestDay, name='bestDay'),
    path('top10Returns', views.top10Returns, name='top10Returns'),
    path('top10Sectors', views.top10Sectors, name='top10Sectors'),
    path('bestPerfStocks', views.bestPerfStocks, name='bestPerfStocks'),
    path('clusterStocks', views.clusterStocks, name='clusterStocks'),
    path('movAvgPredictor', views.movAvgPredictor, name='movAvgPredictor'),
    path('movAvgPredictorMACD', views.movAvgPredictorMACD, name='movAvgPredictorMACD'),
    path('aggloClustering', views.aggloClustering, name='aggloClustering'),
    
]