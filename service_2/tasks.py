from celery import shared_task
from yahoo_fin.stock_info import *
from threading import Thread
import queue

@shared_task(bind = True)
def update_stock(self, stockPicker):
    data = {}
    available_stocks = tickers_sp500()
    for i in stockPicker:
        if i in available_stocks:
            pass
        else:
            stockPicker.remove(i)
    
    n_threads = len(stockPicker)
    thread_list = []
    que = queue.Queue()
    for i in range(n_threads):
        thread = Thread(target= lambda q, arg1: q.put({stockPicker[i]: get_quote_table(arg1)}), args=(que, stockPicker[i]))
        thread_list.append(thread)
        thread_list[i].start()

    for thread in thread_list:
        thread.join()

    while not que.empty():
        result = que.get()
        data.update(result)
