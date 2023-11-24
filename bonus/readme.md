Bonus section (Webserver)

activate venv
./bonus_env/scripts/activate 

cd bonus/controller/bonus_webserver
python manage.py runserver


Bonus section (Webpage)

configure port number of controller in the ./controller/.env file

cd controller
python app.py 

configure port number in .env file of ./view/.env