import numpy as np

#To do: implememnt threshold finding for split method

def get_err(data,thresh):
    """
    Splits the data using the index given by thresh
    Fits a line to each section
    Finds the error of both fits
    Combines them and returns this error
    """

    y_h = data[thresh:]
    y_l = data[:thresh]

    x_h = range(len(y_h))
    x_l = range(len(y_l))

    m_h,c_h = np.polyfit(x_h,y_h,1)
    m_l,c_l = np.polyfit(x_l,y_l,1)

    y_h_fit = x_h*m_h + c_h
    y_l_fit = x_l*m_l + c_l

    err_h = np.sqrt(np.mean(np.square(y_h-y_h_fit)))
    err_l = np.sqrt(np.mean(np.square(y_l-y_l_fit)))

    err = np.asarray([err_h,err_l])

    return np.linalg.norm(err)

def get_highlow_idx(data,num_iters=5):
    """
    Assumes that the data has a fairly obvious split between high and low values
    Sorts the data by value and fits two straight lines in two regions separated by a threshold
    Finds the value of the threshold that minimises the error in the two linear fits
    """
    sorted_idx = np.argsort(data)
    sorted_data = data[sorted_idx[::100]]

    l = len(sorted_data)

    thresh_high = int(0.99*l)
    if (thresh_high == l-1):
        thresh_high -= 1

    thresh_low = int(0.01*l)
    if (thresh_low == 0):
        thresh_low += 1
    
    thresh_mid = int(0.75*l)


    # for i in range(num_iters):

    #     err_high = get_err(sorted_data,thresh_high)
    #     err_low = get_err(sorted_data,thresh_low)
    #     err_mid = get_err(sorted_data,thresh_mid)

    #     if (err_low > err_mid and err_mid > err_high):
    #         thresh_low = thresh_mid
    #         thresh_mid = thresh_high
    #         thresh_high = 

def get_sampled_idx(data,method,kwargs):
    """
    Based on the data and method, the function returns indices for train and test sets
    Data is a float array containing densities
    Allowed methods:
        'value_weighted': rearranges the indices, with the probability of being at the beginning 
            of the set of indices proportional to the absolute value of the data
        'split': splits the data into 'high' and 'low' regions, uniformly sampling from both, 
            the ratio of 'high' and 'low' data in the final set is determined by highlow_ratio
    """

    allowed_methods = ['value_weighted','split']
    default_kwargs = {'value_weighted':{'weighting':'linear','keep_fraction':0.1},\
        'split':{'high_fraction':0.95,'low_fraction':None,'highlow_ratio':0.75,'threshold':0.9*len(data)}}
    
    #highlow ratio is (#high indices/total indices)

    assert (method in allowed_methods), "Invalid method supplied"

    method_args = default_kwargs[method]

    for key in kwargs.keys():
        if (key in method_args.keys()):
            method_args[key] = kwargs[key]

    #the data needs to be rank 1
    #assume that the interesting information is in the 0 axis
    if (isinstance(data,list)):
        #indexing here because the array would otherwise be of shape (1,*), and we want (*,)
        data = np.asarray(data)[0,:]
    elif (len(data.shape)>1):
        data = data[:,0]
    
    if (method == 'value_weighted'):

        if (method_args['weighting']== 'linear'):
            p = np.abs(data)
        elif (method_args['weighting']=='sqrt'):
            p = np.sqrt(np.abs(data))
        elif (method_args['weighting']=='square'):
            p = np.square(data)
        elif (method_args['weighting']=='softplus'):
            p = np.log(1+np.exp(abs(data)))
        else:
            raise NotImplementedError
        p /= np.sum(p)
        rand_idx = np.random.choice(range(len(data)),len(data),replace=False,p=p)

        split_idx = int(len(rand_idx)*method_args['keep_fraction'])
        keep_idx = rand_idx[:split_idx]

    elif (method == 'split'):

        if (method_args['threshold'] is None):
            raise NotImplementedError
            #high_idx,low_idx = get_highlow_idx(data)
        
        else:
            sorted_idx = np.argsort(data)
            threshold = int(len(sorted_idx)*method_args['threshold'])
            high_idx = sorted_idx[threshold:]
            low_idx = sorted_idx[:threshold]

        #randomise order
        high_idx = high_idx[np.random.choice(range(len(high_idx)),len(high_idx),replace=False)]
        low_idx = low_idx[np.random.choice(range(len(low_idx)),len(low_idx),replace=False)]

        #train_ratio is the fraction of the total train data that is from the high region
        r = method_args['highlow_ratio']
        #the quantity r/(1-r), or its inverse, is what's used
        ratio = r/(1-r)

        if (method_args['high_fraction'] is not None and method_args['low_fraction'] is not None):
            method_args['high_fraction'] = None
        if (method_args['low_fraction'] is not None):
            low_thresh = int(method_args['low_fraction']*len(low_idx))
            high_thresh = min([int(low_thresh*ratio),len(high_idx)-1])
        
        if (method_args['high_fraction'] is not None):
            high_thresh = int(method_args['high_fraction']*len(high_idx))
            low_thresh = min([int(high_thresh/ratio),len(low_idx)-1])

        low_train = low_idx[:low_thresh]
        #low_test = low_idx[low_thresh:]

        high_train = high_idx[:high_thresh]
        #high_test = high_idx[high_thresh:]

        keep_idx = np.hstack((high_train,low_train))

    return keep_idx

def get_sampled_trainNtest(data,method='uniform',**kwargs):
    """
    Based on the data and method, the function returns indices for train and test sets
    Data is a float array containing densities
    Allowed methods:
        'uniform': randomly rearranges the data and splits into train and test based on train_fraction
        'value_weighted': rearranges the indices, with the probability of being at the beginning 
            of the set of indices proportional to the absolute value of the data
        'split': splits the data into 'high' and 'low' regions, uniformly sampling from both, 
            the ratio of 'high' and 'low' data in the final training set is determined by train_ratio
    """

    allowed_methods = ['uniform','value_weighted','split']
    default_kwargs = {'uniform':{'train_fraction':0.95},\
        'value_weighted':{'train_fraction':0.2,'weighting':'linear'},\
        'split':{'high_fraction':0.95,'low_fraction':None,'train_ratio':0.75,'threshold':None}}
    
    #train ratio is (#high indices/total indices)

    assert (method in allowed_methods), "Invalid method supplied"

    method_args = default_kwargs[method]

    for key, value in enumerate(kwargs):
        if (value in method_args.keys()):
            method_args[value] = kwargs[value]

    #the data needs to be rank 1
    #assume that the interesting information is in the 0 axis
    if (len(data.shape)>1):
        data = data[:,0]
    
    if (method in ['uniform','value_weighted']):
        if (method == 'uniform'):
            p = None
        else:
            if (method_args['weighting']== 'linear'):
                p = np.abs(data)
            elif (method_args['weighting']=='sqrt'):
                p = np.sqrt(np.abs(data))
            elif (method_args['weighting']=='square'):
                p = np.square(data)
            elif (method_args['weighting']=='softplus'):
                p = np.log(1+np.exp(abs(data)))
            else:
                raise NotImplementedError
            p /= np.sum(p)
        rand_idx = np.random.choice(range(len(data)),len(data),replace=False,p=p)

        split_idx = int(len(rand_idx)*method_args['train_fraction'])
        train_idx = rand_idx[:split_idx]
        test_idx = rand_idx[split_idx:]

    elif (method == 'split'):

        if (method_args['threshold'] is None):
            raise NotImplementedError
            #high_idx,low_idx = get_highlow_idx(data)
        
        else:
            sorted_idx = np.argsort(data)
            threshold = int(len(sorted_idx)*method_args['threshold'])
            high_idx = sorted_idx[threshold:]
            low_idx = sorted_idx[:threshold]

        #randomise order
        high_idx = high_idx[np.random.choice(range(len(high_idx)),len(high_idx),replace=False)]
        low_idx = low_idx[np.random.choice(range(len(low_idx)),len(low_idx),replace=False)]

        #train_ratio is the fraction of the total train data that is from the high region
        r = method_args['train_ratio']
        #the quantity r/(1-r), or its inverse, is what's used
        ratio = r/(1-r)

        if (method_args['high_fraction'] is not None and method_args['low_fraction'] is not None):
            method_args['high_fraction'] = None
        if (method_args['low_fraction'] is not None):
            low_thresh = int(method_args['low_fraction']*len(low_idx))
            high_thresh = min([int(low_thresh*ratio),len(high_idx)-1])
        
        if (method_args['high_fraction'] is not None):
            high_thresh = int(method_args['high_fraction']*len(high_idx))
            low_thresh = min([int(high_thresh/ratio),len(low_idx)-1])

        low_train = low_idx[:low_thresh]
        low_test = low_idx[low_thresh:]

        high_train = high_idx[:high_thresh]
        high_test = high_idx[high_thresh:]

        train_idx = np.hstack((high_train,low_train))
        test_idx = np.hstack((high_test,low_test))

    return train_idx , test_idx

def get_dist(xyz,at_pos,cell):
    """
    Parameters:
        xyz: (array)  shape (N,3). contains the positions of the grid points
        at_pos: (array) shape (3). contains the position of the atom
        cell: (array) shape (3,3). cell vectors
    Returns:
        min_dist: (array) shape (N,3). the vector corresponding to the smallest
            distance between the grid coordinate and the atom, accounting for 
            the periodic unit cell
    """
    dist = xyz-at_pos

    dist_arr = np.zeros((27,xyz.shape[0]))
    cntr = 0
    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                coeffs = np.array([i,j,k])
                dist_arr[cntr,:] = np.linalg.norm(dist+np.dot(coeffs,cell),axis=1)
                
                
                cntr += 1
    min_dist = np.min(dist_arr,axis=0)

    return min_dist
    

def get_points_near_atoms(xyz,at_posns,cell,max_dist):
    """
    Parameters:
        xyz: (array) shape (N,3). contains the positions of the grid points
        at_posns: (array) shape (n,3). contains the positions of the atoms
        cell: (array) shape (3,3). cell vectors
        max_dist: (float) maximum allowed distance a grid point can have from an atom
    Returns:
        allowed_idx: (list) contains the indices of all of the grid points within the cutoff distance from an atom
    """

    assert (max_dist<0.5*min(np.linalg.norm(cell,axis=1))), "All points fit in max dist, are you sure you need this function?"

    #generate allowed spheres for each atom
    allowedlist = []
    
    for i in range(at_posns.shape[0]):
        at_pos = at_posns[i,:]
        dists = get_dist(xyz,at_pos,cell)

        condition = np.zeros(xyz.shape[0]).astype(bool)
        condition = (np.abs(dists)<max_dist)
        allowed_idx = np.where(condition)[0]

        #allowed_idx = xyz[dists[:,0]<max_dist & dists[:,1]<max_dist & dists[:,2]<max_dist]
        diff = set(allowed_idx)-set(allowedlist)
        allowedlist = allowedlist + list(diff)
    
    return allowedlist
    
    



