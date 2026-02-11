# Two_Stage_Cascade
Here we consider a two-stage dark matter decay chain (A->2B), where the decay is triggered by spontaneous symmetry breaking in the dark sector. Here A has an equation of state parameter of zero, whereas B has an equation of state parameter of -1/3. This results in a larger late-time value for the Hubble parameter, allowing for partial alleviation of the Hubble tension. Note that A and B are a subcomponent of the dark matter. 

Grid_Search.py: This performs a grid search over the parameter space to find the best combination. This values are then used as the initial values for Powell's method. 

Powell_Fit.ipynb: This uses Powell's method to fit parameters (decay rate, redshift at symmetry breaking, and fraction of dark matter living in the cascade at z=1090) to measured H(z).

Genetic_Algorithm_Parameter_Fit.py: This uses a genetic algorithm to fit parameters (decay rate, redshift at symmetry breaking, and fraction of dark matter living in the cascade at z=1090) to measured H(z).

Results_Plotter.ipynb: This takes the best-fit parameters from the two methods used and plots both H(z) and the fractional change in H(z) as compared to Lambda-CDM. 
