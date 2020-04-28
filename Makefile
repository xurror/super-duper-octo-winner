init:
	python -m venv venv
	pip install -r requirements.txt

test:
	nosetests tests