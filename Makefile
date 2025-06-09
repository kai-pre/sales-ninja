.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y salesninja || :
	@pip install -e .

# run_preprocess:
# 	python -c 'from salesninja.interface.main import preprocess; preprocess()'

# run_train:
# 	python -c 'from taxifare.interface.main import train; train()'

# run_pred:
# 	python -c 'from taxifare.interface.main import pred; pred()'

# run_evaluate:
# 	python -c 'from taxifare.interface.main import evaluate; evaluate()'

# run_all: run_preprocess run_train run_pred run_evaluate

run_api:
	uvicorn salesninja.api.api:app --reload --port 8899

default:
	pass
