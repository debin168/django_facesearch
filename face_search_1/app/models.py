from __future__ import unicode_literals

from django.db import models

# import os,django
#
#
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'face_search_1.face_search_1.settings')
#
#
#
# django.setup()

# Create your models here.
class Image(models.Model):
    uid=models.CharField(max_length=50)
    path=models.FileField(upload_to = './upload/')

    def __unicode__(self):
        return self.uid