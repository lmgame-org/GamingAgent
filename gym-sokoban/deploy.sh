python setup.py sdist bdist_wheel

# To production: twine upload dist/*
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*