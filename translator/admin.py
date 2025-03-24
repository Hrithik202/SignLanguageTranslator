from django.contrib import admin

from .models import ProcessedImage
admin.site.register(ProcessedImage)

from .models import SignLanguageInput
admin.site.register(SignLanguageInput)

# Register your models here.
