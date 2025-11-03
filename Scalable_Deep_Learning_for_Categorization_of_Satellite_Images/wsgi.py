"""
WSGI config for Scalable_Deep_Learning_for_Categorization_of_Satellite_Images project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Scalable_Deep_Learning_for_Categorization_of_Satellite_Images.settings')

application = get_wsgi_application()
