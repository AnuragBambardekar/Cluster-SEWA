from __future__ import absolute_import, unicode_literals
import os

from celery import Celery
from django.conf import settings
# from celery.schedules import crontab # to schedule tasks at specific times

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'clusterapp.settings')

app = Celery('clusterapp')
app.conf.enable_utc = False
app.conf.update(timezone = 'America/New_York')

app.config_from_object(settings, namespace='CELERY')

app.conf.beat_schedule = {
    # 'every-10-seconds' : {
    #     'task': 'service_2.tasks.update_stock',
    #     'schedule': 10,
    #     'args': (['AAPL', 'MSFT'],)
    # },
}

app.autodiscover_tasks()

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')