import numpy as np
from features.regressor import regressor
import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.rc('text', usetex=True)
plt.rc('font',family='seriff')
rcParams['figure.dpi'] = 150

def get_rmse(net,x,y):
    y_pred, _ = net.predict(x)
    rmse = np.sqrt(np.sum(np.square(y-y_pred)))
    return rmse

def parity_plot(net,x,y,title_input=None,print_RMSE=True,save=True,savename='parity.pdf'):
    """
    Given a network and some test data (here it's x and y), plot predicted versus accurate spin density 
    """
    savename = '{}_{}'.format(title_input,savename)
    y_pred, _ = net.predict(x)

    minmax = np.array([np.min(y),np.max(y)])

    
    plt.plot(minmax,minmax,'r')
    plt.plot(y,y_pred,'b.',markersize=2)
    plt.xlabel(r'Spin Density ($\hbar$/2)')
    plt.ylabel(r'Predicted Spin Density ($\hbar$/2)')
    if (title_input is not None):
        plt.title('Spin Density Parity Plot for {} Data'.format(title_input))
    if (save):
        plt.savefig(savename)
    else:
        plt.show()
    plt.close()

    if (print_RMSE):
        print(get_rmse(net,x,y))

def rmse_plot(net,log_y=False,save=False,savename='RMSEplot.pdf'):
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
        if (log_y):
            plt.semilogy(x,rmse[i,:])
            plt.semilogy(x_val,rmse_val[i,:])
        else:
            plt.plot(x,rmse[i,:])
            plt.plot(x_val,rmse_val[i,:])
        legend.append('Net {} Train'.format(i+1))
        legend.append('Net {} Val'.format(i+1))
    plt.legend(legend)
    plt.xlabel("Percentage Completion of Training")
    plt.ylabel("RMSE")
    plt.title("Network Training and Validation RMSE During Training")
    if (save):
        plt.savefig(savename)
    else:
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
