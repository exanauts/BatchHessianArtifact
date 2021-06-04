
JULIA_EXEC = julia
PYTHON_EXEC = python

instantiate: 
	$(JULIA_EXEC) --project -e "using Pkg; Pkg.instantiate()"

run: 
	$(JULIA_EXEC) --project -e "include(\"sc2021.jl\")"

plot: 
	$(PYTHON_EXEC) figures/batch_hessian.py 
	$(PYTHON_EXEC) figures/batch_hess_decomposed.py 

