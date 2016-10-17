import os.path 
import numpy as np

def lreg(x,y):
    #execute a robust linear regression isong RANSAC method
    from sklearn import linear_model
    model = linear_model.LinearRegression()
    model.fit(x, y)

    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
    model_ransac.fit(x, y)

    return model_ransac

def repair_1(x,y):
    '''this function reparis a prediction when the datum to the left and right are the same
    In that case is highly probable the datum should be the same as well
    '''
    n = len(x)

    k = 0
    if n >=3:
        for i in xrange(2,n-1):
            if y[i-1] == y[i+1] and y[i-1] != y[i]:
                y[i] = y[i-1]
                k += 1
    return k

def repair_2(x,y,debug=False):
    '''this function reparis a prediction at breaks sections.
    for example: 5 5 5 5 4 4 4 6 6 
    Under the assumtion tag are increasing,  if there is one that is going down we should fix it.
    Three cases:
    a) to the right there is a value less than to the left, therefore must be the same value as the left
    b) to the right there is a value greater than to the left, therefore we choose the closest.
    c) to the right there is a value equal, the same applied as a)
    '''
    n = len(x)

    k = 0
    if n >=3:
        #look for the first 6
        i_start = 2
        for i in xrange(2,n-1):
            if y[i] == 6:
                i_start = i
                break
        #9 11 10
        if debug: print "i_start",i_start
        for i in xrange(i_start+1,n-1):
            if debug: print "(i-1|i|i+1)",i,x[i],":",y[i-1],y[i],y[i+1]
            if y[i] < y[i-1]:
                k += 1
                if y[i-1] >= y[i+1]: #make it equal to i-1
                    if debug: print "changing",y[i],"by",y[i-1]
                    y[i] = y[i-1]
                else:
                    if (x[i] - x[i-1]) <= (x[i+1] - x[i]): #closet to left
                        if debug: print "changing",y[i],"by",y[i-1]
                        y[i] = y[i-1]
                    else:
                        if debug: print "changing",y[i],"by",y[i+1]
                        y[i] = y[i+1]
                        
            elif y[i] > y[i-1]:
                if y[i-1] == y[i+1]: #neighbors are the same
                    k += 1
                    if debug: print "changing",y[i],"by",y[i-1]
                    y[i] = y[i-1]
                elif y[i-1] < y[i+1] and y[i] > y[i+1]: #neighbors are increasingly consisten
                    if (x[i] - x[i-1]) <= (x[i+1] - x[i]): #closet to left
                        if debug: print "changing",y[i],"by",y[i-1]
                        y[i] = y[i-1]
                    else:
                        if debug: print "changing",y[i],"by",y[i+1]
                        y[i] = y[i+1]

        #the last one
        if y[-1] < y[-2]:
            if debug: print "changing",y[-1],"by",y[-2]
            y[-1] = y[-2]
        
    return k


def fix_prediction(images_numbers, predictions):
    x = np.int32(images_numbers)
    y = np.int32(predictions)

    n = len(x)

    while True:
        k = repair_1(x,y)
        if k == 0: break

    #look from 7 to 53
    i_start = 1
    for i in xrange(n):
        if y[i] == 7:
            i_start = i
            break
    th = 6
    

    i_end = n-1
    for i in xrange(n-1,0,-1):
        if y[i] == 53:
            i_end = i
            break
    
    X = x[i_start:i_end]
    m = len(X)
    X = X.reshape((m,1))
    model = lreg(X,y[i_start:i_end])

    pred_ori = model.predict(X)
    
    pred = np.array(y)
    pred[i_start:i_end] = pred_ori

    pred = np.int32(np.clip(pred,th,54))

    #difference
    diff = np.abs(y-pred)
    #find higher
    indices = np.where(diff>= th)

    y2 = np.array(y)
    y2[indices] = pred[indices]

    k = repair_2(x,y2,debug=True)

    return y2
