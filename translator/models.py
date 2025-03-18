from django.db import models

class SignLanguageInput(models.Model):
    file = models.FileField(upload_to='sign_language_inputs/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Upload {self.id} at {self.uploaded_at}"


# Create your models here.
