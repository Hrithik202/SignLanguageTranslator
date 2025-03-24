from django.db import models

class SignLanguageInput(models.Model):
    image = models.ImageField(upload_to='sign_language_inputs/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Upload {self.id} at {self.uploaded_at}"


class ProcessedImage(models.Model):
    original_image = models.ForeignKey('translator.SignLanguageInput', on_delete=models.CASCADE)  
    processed_image = models.ImageField(upload_to='processed_images/')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Processed Image {self.id} from Upload {self.original_image.id}"
