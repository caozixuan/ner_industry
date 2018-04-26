# -*- coding: utf-8 -*-
# Generated by Django 1.10.3 on 2018-04-20 11:25
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Word',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('sentence_id', models.IntegerField()),
                ('name', models.CharField(max_length=100)),
                ('div_type', models.IntegerField(default=0, null=True)),
            ],
        ),
    ]
