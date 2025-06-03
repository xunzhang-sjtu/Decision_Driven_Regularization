# This file contains information about DDR

## DDR application
### Benchmarks
- File: [Benchmarks.ipynb](/DDR/Benchmarks.ipynb)

- This file includes:
    - Figure 1: Fidelity peroformance
    - Figure 2: Performance of DDR against OLS under baseline setting
    - Figure 3: Calibration of $\lambda$ and $\mu$


### Figure 4: Dimensions, predictors, sample size and noise.
- File: [Figure4.ipynb](/DDR/Figure4.ipynb)

### Figure 5: Model misspecification.
- File: [Figure5.ipynb](/DDR/Figure5_Model_Mis.ipynb)
- <font color="red"> The regret and head-to-head shown in Original Figure 5 is inconsistent with the results presented in Figure 2. </font>

### Figure 6: With constraint
- File: [Figure6_Further_Structure.ipynb](/DDR/Figure6_Further_Structure.ipynb)
- <font color="red"> In the original implementation, although the ddr_solver function was invoked, the corresponding dual constraints $Ay\leq b$ was not included in the formulation.  </font>
- <font color="red"> Why do we need to include the threshold in the formulation? </font>