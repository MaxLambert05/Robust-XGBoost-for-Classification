# Robust-XGBoost-for-Classification
This repository contains the most relevant Python code that was used in Robust XGBoost for Binary Classification. The core file contains all the fundamentals of the simulations, including the
definitions of the objective functions to be implemented in XGBoost. The other scripts that import the core file is where parameters and different simulation settings are to be specified.

Recommended Flow:
1) Simulation run with FlipInTheMargin_Config_LogReg_Github.py (output as .xlsx or .csv file)
2) Format the file with format_summary_Final_Github.py (output as .xlsx file)
3) Plot metric evolution with PlotPerMetric_Github.py or Write LaTeX table (\usepackage{booktabs}) for Overleaf with table_summary_Github.py
