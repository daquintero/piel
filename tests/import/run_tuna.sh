python -X importtime test_import_core.py 2> import_core.log
python -X importtime test_import_experimental.py 2> import_experimental.log
tuna import_core.log
tuna import_experimental.log
