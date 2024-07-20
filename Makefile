install:
	bash scripts/install.sh

download-pretrained-models:
	bash scripts/download_pretrained_models.sh $(DIR)

download-results:
	bash scripts/download_results.sh $(DIR)

run:
	bash scripts/run.sh $(CONFIG)

run-clr:
	bash scripts/run_clr.sh $(CONFIG)


run-bcp:
	bash scripts/run_bcp.sh $(CONFIG)

distributed_run:
	bash scripts/distributed_run.sh $(CONFIG) $(NGPU)

