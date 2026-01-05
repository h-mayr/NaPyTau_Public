# File Responsibilities

### time.py
This file contains service functions for the polynomial evaluation. Since the "DataSet", that is handed over to the functions in [`polynomials.py`](#polynomialspy), only contains measuring distances, they have to be converted to times, using the relative velocity. For this step, two functions exist in this file, one to calculate a single time and one to calculate a whole array of times.

### polynomials.py
This file provides two functionalities. The first is to evaluate polynomials at measuring times. One function evaluates a polynomial function directly at given measuring times and the other takes the derivative of a polynomial function and then evaluates it at the given measuring times. The coefficients of the polynomial functions are expected to be provided in increasing order of degree, e.g. the polynomial $2x^2+4x+3$ is expected to be provided as $[3, 4, 2]$.

The other functionality is to calculate polynomial coefficients. One function just calculates them for the standard fit and the other calculates them for a fit that takes a specific tau factor into account.

### chi.py
This file contains one function to calculate $\chi^{2}$ via this formula:

\[
\chi^{2} = \sum_{i}((\frac{I^{sh}_{i}-f^{(a_{1}, ..., a_{n})}(t_{i})}{\Delta I^{sh}_{i}})^2+w(\frac{I^{us}_{i} - \tilde{t}^{hyp}\frac{d}{dt}f^{(a_{1}, ..., a_{n})}(t_{i})}{\Delta I^{us}_{i}})^2) 
\]

and another function to optimize the tau factor $\tilde{t}^{hyp}$ by minimizing $\chi^{2}$. This optimization is implemented by using the minimize function from "scipy.optimize" with the mean of the total possible range of the tau factor as a starting point.

### tau.py
This file provides functionality to calculate the lifetime $\tau_{i}$ via this formula:

$$
\tau_{i} = \frac{I^{us}_{i}}{\frac{d}{dt}f^{opt}(t_{i})}
$$

with $f^{opt}(t_{i})$ being the fit function for the optimal $\tilde{t}^{hyp}$, i.e. $\tilde{t}^{opt}$.

### delta_tau.py
Here we handle the error calculation for $\Delta\tau_{i}^{2}$. It follows this formula:

$$
\Delta\tau_{i}^{2} = \frac{\Delta I^{us2}_{i}}{\dot{P}^{2}_{j(i)}} + \frac{I^{us2}_{i}}{\dot{P}^{4}_{j(i)}} \Delta\dot{P}^{2}_{j(i)} + \frac{I^{us}_{i}\tilde{t}}{\dot{P}^{3}_{j(i)}} \Delta\dot{P}^2_{j(i)}
$$

As a part of the error calculation we also need the covariance matrix, for the calculation of which a function is provided.

### tau_final.py
This file merges the lifetime $\tau_{i}$ and the error $\Delta\tau_{i}$ to calculate the weighted mean $\tau_{final}$.

### core.py
This file acts as an interface to the other modules. It provides three functions that are intended to be called by the GUI and Headless modules for the tau factor and lifetime calculations.

The "calculate_lifetime_for_fit" function automatically fits the function for the default tau factor and calculates the polynomial coefficients for that fit. It then calculates and returns the corresponding lifetime.

The "calculate_optimal_tau_factor" function also fits the function and calculates the polynomial coefficients and then calculates the optimal tau factor for this fit.

The "calculate_lifetime_for_custom_tau_factor" function directly calculates the polynomial coefficients for a custom tau factor that can be set via the slider in the GUI, without fitting. It then uses this polynomial to calculate the lifetime.