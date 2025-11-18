JULIA:=julia

default: help

setup:
	${JULIA} -e 'import Pkg; Pkg.add(["JuliaFormatter"])'

format:
	${JULIA} -e 'using JuliaFormatter; format(".")'

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