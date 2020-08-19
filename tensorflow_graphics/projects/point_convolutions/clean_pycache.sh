find . -name "*.pyc" -exec rm -f {} \;
find . -name "__pycache__" -exec rm -r {} \;