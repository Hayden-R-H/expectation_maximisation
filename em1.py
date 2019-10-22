"""
:@title: Expectation Maximisation Fisher Example
:@author: Hayden Hohns
:@date: 21/09/2019
:@brief: This Python code is based on a classic example by the statistician 
Sir Ronald Aylmer Fisher. 
"""

import matplotlib.pyplot as plt
import numpy as np
import sys


def em_fisher(y_1, y_2, y_3, y_4, tol=10e-6, start=0.5):
    """
    BRIEF

    This function defines the problem and includes nested functions for the 'E' 
    and 'M' steps. 

    ARGUMENTS

    :@y_1:
    :@y_2:
    :@y_3:
    :@y_4:
    :@tol: A float that defines the tolerance for the terminating condition of 
    the algorithm.
    :@start: An initial guess for the model (also a float).

    RETURNS

    :@psi:

    """


    def e_step(psi_c, y_1):
        y_11 = (y_1 / 2) / (1 / 2 + psi_c / 4)
        y_12 = y_1 - y_11
        return y_11, y_12


    def m_step(y_12, y_11, y_4, n):
        psi_new = (y_12 + y_4) / (n - y_11)
        return psi_new
    

    psi_last = 0
    n = y_1 + y_2 + y_3 + y_4 
    psi_current = start
    psi = psi_current
    estimates = []
    numIter = 0

    while np.abs(psi_last - psi) > tol:
        y_11, y_12 = e_step(psi_current, y_1)
        psi = m_step(y_12, y_11, y_4, n)
        psi_last = psi_current
        psi_current = psi
        numIter += 1
        estimates.append(psi_current)

    return estimates, numIter


def make_plot(markovChain, numIter: int):
    """
    BRIEF


    ARGUMENTS

    :@markovChain: A list containing the value of the parameter estimates.
    :@numIter: An integer representing the number of iterations it took to get to the current/latest parameter value.

    RETURNS


    """


    f, ax = plt.subplots(1, 1, dpi=100, figsize=(16, 10))

    # Format spines and ticks
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.plot(markovChain, linestyle='', marker='o')
    ax.set_title('Expectation Maximisation Algorithm', fontsize='20')
    ax.set_ylabel('Parameter Value', fontsize='14')
    plt.xlabel('Iterations = ' + str(numIter), fontsize='14')

    f.savefig('em_fisher.png', bbox_inches='tight')


def main(tolerance=1e-6, start=0.5):
    """
    BRIEF

    ARGUMENTS

    RETURNS

    """

    y = [125.0, 18.0, 20.0, 34.0]
    start = 0.1
    estimates, numIter = em_fisher(*y, tol=1e-9, start=start)
    make_plot(estimates, numIter)
    
    print("The number of iterations required for convergence: " + str(numIter))

    return 0

if __name__ == "__main__":
    main()