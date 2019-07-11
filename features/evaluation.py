import numpy as np
from features.regressor import regressor
import matplotlib.pyplot as plt

def parity_plot(net,x,y,title_input):
    """
    Given a network and some test data (here it's x and y), plot predicted versus accurate spin density 
    """
    y_pred, _ = net.predict(x)

    minmax = np.array([np.min(y),np.max(y)])

    plt.plot(y,y_pred,'b.')
    plt.plot(minmax,minmax,'r')
    plt.xlabel('Spin Density')
    plt.ylabel('Predicted Spin Density')
    plt.title('Spin Density Parity Plot for {} Data'.format(title_input))
    plt.show()
    plt.close()

def rmse_plot(net):
    """
    For a trained network, plots the rmse for the training and validation set over the course of a training run
    """
    rmse_val = net.rmse_val
    rmse = net.rmse

    #rmses are arrays of shape (Nensemble,#points,1)
    #remove the last index and iterate over the first

    x_val = np.linspace(0,100,rmse_val.shape[1])
    x = np.linspace(0,100,rmse.shape[1])

    legend = []
    for i in range(rmse.shape[0]):
        print(min(rmse_val[i,:]))
        plt.plot(x,rmse[i,:])
        plt.plot(x_val,rmse_val[i,:])
        legend.append('Net {} Train'.format(i+1))
        legend.append('Net {} Val'.format(i+1))
    plt.legend(legend)
    plt.xlabel("Percentage Completion of Training")
    plt.ylabel("RMSE")
    plt.title("Network Training and Validation RMSE During Training")
    plt.show()
    plt.close()

def Ensemble_comparison(net):
    val_x = net.train_data.xs_val
    val_y = net.train_data.ys_val

    means = net.predict_individual_values(val_x)

    for i in range(net.Nensemble):
        rmse = np.sqrt(np.mean(np.square(means[i,:,0]-val_y)))
        print('Network {} RMSE: {}'.format(i+1,rmse))
    return
