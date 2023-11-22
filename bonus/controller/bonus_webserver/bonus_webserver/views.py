from django.http import JsonResponse

def api_home(request):
    data = {
        "message": "Hello, this is a simple API endpoint!"
    }
    return JsonResponse(data)
