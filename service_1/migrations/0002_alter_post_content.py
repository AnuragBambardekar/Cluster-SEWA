# Generated by Django 4.1.6 on 2023-05-04 17:36

from django.db import migrations
import tinymce.models


class Migration(migrations.Migration):

    dependencies = [
        ('service_1', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='post',
            name='content',
            field=tinymce.models.HTMLField(),
        ),
    ]
