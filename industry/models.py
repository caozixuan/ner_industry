from __future__ import unicode_literals

from django.db import models

# Create your models here.

class Word(models.Model):
    sentence_id = models.IntegerField(null=False)
    name=models.CharField(max_length=100)
    div_type=models.IntegerField(null=True,default=0)


class WordTest(models.Model):
    sentence_id = models.IntegerField(null=False)
    name=models.CharField(max_length=100)
    div_type=models.IntegerField(null=True,default=0)


