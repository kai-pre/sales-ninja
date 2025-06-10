.DEFAULT_GOAL := default
SALESNINJA_QUERIES=~/.salesninja/queried/
SALESNINJA_PROCESSED=~/.salesninja/processed/
SALESNINJA_ML=~/.salesninja/training_outputs/
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y salesninja || :
	@pip install -e .

run_preprocess:
	python -c 'from salesninja.dashboard.main import preprocess; preprocess()'

run_train:
	python -c 'from salesninja.dashboard.main import train; train()'

run_pred:
	python -c 'from salesninja.dashboard.main import predict; predict()'

run_evaluate:
	python -c 'from salesninja.dashboard.main import evaluate; evaluate()'

run_all: run_preprocess run_train run_pred run_evaluate

run_api:
	uvicorn salesninja.api.api:app --reload --port 8899

default:
	pass

remove_local_files:
	rm -rf ${SALESNINJA_ML}
	rm -rf ${SALESNINJA_PROCESSED}
	rm -rf ${SALESNINJA_QUERIES}
