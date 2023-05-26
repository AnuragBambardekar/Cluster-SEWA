from django.contrib import admin
from .models import Category,Post

# Register your models here.

# For configuration of Category - Admin
class CategoryAdmin(admin.ModelAdmin): #Encapsulate Admin options
    # Which all options to show
    list_display = ('image_tag','title','description','url','add_date')
    search_fields = ('title',)
    list_filter = ('title',)

# For configuration of Category - Admin
class PostAdmin(admin.ModelAdmin): #Encapsulate Admin options
    # Which all options to show
    list_display = ('image_tag','title','url')
    search_fields = ('title',)
    list_filter = ('cat',)
    list_per_page = 5

    class Media:
        js=('https://cdn.tiny.cloud/1/no-api-key/tinymce/6/tinymce.min.js','js/script.js',) #this is a tuple

admin.site.register(Category, CategoryAdmin)
admin.site.register(Post, PostAdmin)