import numpy as np
import matplotlib.pyplot as plt

def init_centers(k=3):
    """
    Randomly initialize cluster centers
    """
    return np.vstack([np.random.uniform(low=2., high=4.5, size=k),
                      np.random.uniform(low=1, high=4.5, size=k)]).T

def k_means(num_clusters, X, centers, update_assignments, 
            update_parameters, n_iter=4):
    """
    Runs the k-means algorithm for n_iter iterations and plots
    the results.
    
    Parameters
    ----------
    num_clusters : int
        The number of disjoint clusters (i.e., k) to search for
        in X
    
    X : numpy array of shape (m, 2)
        An array of m data points in R^2
        
    centers : numpy array of shape (num_clusters, 2)
        The coordinates for the centers of each cluster
        
    update_assignments : function
        The function you completed in part A
    
    update_parameters : function
        The function you completed in part B
    
    n_iter : int (optional)
        The number of iterations to run the k_means update procedure.
        If not specified, defaults to 4.
        
    Returns
    -------
    cluster_assignments : numpy array of shape (m,) 
        The final label assignments for each of the data points in X
        
    centers : numpy array of shape (num_clusters, 2)
        The final cluster centroids in R^2 after running k-means
    """
    fig, ax = plt.subplots(2, n_iter)
    fig.tight_layout(h_pad=2.0)
    ax = ax.flatten()

    for i in range(n_iter):
        # Step 1: Update cluster assignments
        cluster_assignments = \
            update_assignments(num_clusters, X, centers)
            
        # plot data with colors corresponding to cluster assignments
        for j in range(X.shape[0]):
            if cluster_assignments[j] == 0:
                ax[2*i].plot(X[j,0], X[j,1], 'r.')
            elif cluster_assignments[j] == 1:
                ax[2*i].plot(X[j,0], X[j,1], 'b.')
            else:
                ax[2*i].plot(X[j,0], X[j,1], 'g.')
                
        # plot the centers as stars with the associated color
        ax[2*i].plot(centers[0,0], centers[0,1], 'r*', markersize=10)
        ax[2*i].plot(centers[1,0], centers[1,1], 'b*', markersize=10)
        ax[2*i].plot(centers[2,0], centers[2,1], 'g*', markersize=10)
        ax[2*i].set_title('Step 1: \nIteration ' + str(i+1))

        ax[2*i].set_xlim([2, 4.5])
        ax[2*i].set_ylim([1, 4.5])
        ax[2*i].set_xticks([]) 
        ax[2*i].set_yticks([])
        
        # Step 2: Update the cluster centers
        centers = \
            update_parameters(num_clusters, X, cluster_assignments)
        
        # Plot data assignments with the updated center positions
        for j in range(X.shape[0]):
            if cluster_assignments[j] == 0:
                ax[2*i+1].plot(X[j,0], X[j,1], 'r.')
            elif cluster_assignments[j] == 1:
                ax[2*i+1].plot(X[j,0], X[j,1], 'b.')
            else:
                ax[2*i+1].plot(X[j,0], X[j,1], 'g.')
                
        # Plot cluster centers as stars
        ax[2*i+1].plot(centers[0][0], 
                       centers[0][1], 
                       'r*', markersize=10)
        ax[2*i+1].plot(centers[1][0], 
                       centers[1][1], 
                       'b*', markersize=10)
        ax[2*i+1].plot(centers[2][0], 
                       centers[2][1], 
                       'g*', markersize=10)
        ax[2*i+1].set_title('Step 2: \nIteration ' + str(i+1))

        ax[2*i+1].set_xlim([2, 4.5])
        ax[2*i+1].set_ylim([1, 4.5])
        ax[2*i+1].set_xticks([])
        ax[2*i+1].set_yticks([])
        
    plt.show()
    return cluster_assignments, centers


def plot_final(X, cluster_assignments, updated_centers, new_object,
               assign_new_object):
    """
    Categorizes a new object and plots it against the true cluster
    labels.
    
    Parameters
    ----------    
    X : numpy array of shape (m, 2)
        An array of m data points in R^2
    
    cluster_assignments : numpy array of shape (m,) 
        The final label assignments for each of the data points in X
    
    updated_centers : numpy array of shape (num_clusters, 2)
        The coordinates for the centers of each cluster after 
        running k_means
        
    new_object : numpy array of shape (2,)
        The (x,y) coordinates of a new object to be classified
    
    assign_new_object : function
        The function you completed in part D    
    """
    fig, ax = plt.subplots(1,2)
    
    # plot data with colors corresponding to cluster assignments
    for j in range(X.shape[0]):
        if cluster_assignments[j] == 0:
            ax[0].plot(X[j,0], X[j,1], 'r.')
        elif cluster_assignments[j] == 1:
            ax[0].plot(X[j,0], X[j,1], 'b.')
        else:
            ax[0].plot(X[j,0], X[j,1], 'g.')
    
    # Generate a label for the new object
    label = assign_new_object(new_object, updated_centers)
    
    # Plot the new object as as big circle on the plot
    if label == 0:
        ax[0].plot(new_object[0], new_object[1], 'ro', markersize=10)
    elif label == 1:
        ax[0].plot(new_object[0], new_object[1], 'bo', markersize=10)
    else:
        ax[0].plot(new_object[0], new_object[1], 'go', markersize=10)
    
    ax[0].set_aspect('equal', 'datalim')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_title('Final Cluster Assignments')
    
    # Plot the true cluster assignments for comparison
    ax[1].plot(X[:10, 0], X[:10, 1], 'r.', label='cats')
    ax[1].plot(X[10:20,0], X[10:20, 1], 'b.', label='dogs')
    ax[1].plot(X[20:, 0], X[20:, 1], 'g.', label='mops')
    ax[1].set_title('True Cluster Assignments')
    ax[1].set_aspect('equal', 'datalim')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].legend()
    plt.show()
