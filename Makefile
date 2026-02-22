JULIA:=julia

default: help

setup:
	${JULIA} --project=@runic --startup-file=no -e 'using Pkg; Pkg.add("Runic")'

format:
	${JULIA} --project=@runic --startup-file=no -e 'using Runic; exit(Runic.main(ARGS))' -- --inplace .

test:
	${JULIA} --project -e 'using Pkg; Pkg.update(); Pkg.test()'

all: setup format test

help:
	@echo "The following make commands are available:"
	@echo " - make setup: install the dependencies for make command"
	@echo " - make format: format codes with JuliaFormatter"
	@echo " - make test: run the tests"
	@echo " - make all: run every commands in the above order"

.PHONY: default setup format test all help