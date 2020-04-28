init:
	python -m venv venv
	pip install -r requirements.txt
	
start:
	python src/main.py

test:
	nosetests tests