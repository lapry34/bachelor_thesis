# Funzione di Goldstein e Price (n=2)
import numpy as np

def goldstein_price_function(x):
    # Function to evaluate the Goldstein-Price function
    c = x[0] + x[1] + 1.0
    d = 19.0 - 14.0 * x[0] + 3.0 * x[0]**2 - 14.0 * x[1] + 6.0 * x[0] * x[1] + 3.0 * x[1]**2
    e = 2.0 * x[0] - 3.0 * x[1]
    f = 18.0 - 32.0 * x[0] + 12.0 * x[0]**2 + 48.0 * x[1] - 36.0 * x[0] * x[1] + 27.0 * x[1]**2

    a = 1.0 + c**2 * d
    b = 30.0 + e**2 * f

    goldstein_price = a * b

    return goldstein_price

def goldstein_price_gradient(x):
    # Function to compute the gradient of the Goldstein-Price function
    c = x[0] + x[1] + 1.0
    d = 19.0 - 14.0 * x[0] + 3.0 * x[0]**2 - 14.0 * x[1] + 6.0 * x[0] * x[1] + 3.0 * x[1]**2
    e = 2.0 * x[0] - 3.0 * x[1]
    f = 18.0 - 32.0 * x[0] + 12.0 * x[0]**2 + 48.0 * x[1] - 36.0 * x[0] * x[1] + 27.0 * x[1]**2

    a = 1.0 + c**2 * d
    b = 30.0 + e**2 * f

    g1 = (2.0 * c * d + c**2 * (-14.0 + 6.0 * x[0] + 6.0 * x[1])) * b + a * (
            4.0 * e * f + e**2 * (-32.0 + 24.0 * x[0] - 36.0 * x[1]))

    g2 = (2.0 * c * d + c**2 * (-14.0 + 6.0 * x[0] + 6.0 * x[1])) * b + a * (
            -6.0 * e * f + e**2 * (48.0 - 36.0 * x[0] + 54.0 * x[1]))

    gradient = np.array([g1, g2])

    return gradient

if __name__ == "__main__":
    # Example usage
    x_example = np.zeros(2)

    function_value = goldstein_price_function(x_example)
    gradient_value = goldstein_price_gradient(x_example)

    print("Function value:", function_value)
    print("Gradient value:", gradient_value)
