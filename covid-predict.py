import numpy as np
import matplotlib.pyplot as plt
from linear_regression import SimpleLinearRegression
from theta_generator import GradientDescent, ThetaGenerator


if __name__ == "__main__":

    print('LOADING DATA...', end='')
    dataset = np.loadtxt('vn-covid-data.csv', delimiter=',')
    print('COMPLETE')

    print('INITIALISE MACHINE...', end='')
    machine = SimpleLinearRegression(dataset, False, GradientDescent(0.02, 500))
    print('COMPLETE')
    
    print('LEARNING...', end='')
    machine.learn()
    print('COMPLETE')
    print('THETA = ' + str(machine.theta))
    print('COMPUTE COST = %.2f' % ThetaGenerator.compute_cost(machine.input_set, machine.output_set, machine.theta), end='')
    print('(ALPHA = %.2f, NUMBER OF ITERATE = %d)' % (machine.theta_generator.alpha, machine.theta_generator.iterate_num))

    print('DISPLAY PLOT:')
    print("\tFIGURE 1: DISPLAY RESULT")
    plt.figure(1)
    plt.plot(machine.input_set[:,1], machine.output_set, 'rx')
    plt.plot(machine.input_set[:,1], machine.predict(machine.input_set), '-b')
    plt.ylabel('Deceased cases')
    plt.xlabel('Confirmed cases')
    plt.suptitle('Number of new deceased cases base on number of new confirmed cases\n by COVID-19 in Vietnam')
    
    print("\tFIGURE 2: LEARNING PROGRESS")
    plt.figure(2)
    plt.plot(machine.theta_generator.cost_history[:,0], machine.theta_generator.cost_history[:,1], '-r')
    plt.xlabel('Time')
    plt.ylabel('Compute cost')
    plt.suptitle('Learning progress')
    plt.show()

