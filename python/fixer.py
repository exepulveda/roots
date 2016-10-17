import os.path 
import numpy as np

def lreg(x,y, th):
    #execute a robust linear regression using RANSAC method
    from sklearn import linear_model
    model = linear_model.LinearRegression()
    model.fit(x, y)

    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(),residual_threshold=th)
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

def detect_fix_outliers(x,y_original,th_detect=3,th_fix=2,debug=False):
    n = len(x)
    X = x.reshape((n,1))
    y = np.array(y_original)


    model = lreg(X,y, th_detect)
    pred = model.predict(X)
    pred = np.int32(np.clip(np.round(pred),6,54))
    line_X = np.arange(np.min(x),np.max(x))
    line_y = model.predict(line_X[:, np.newaxis])

    # find ransac outliers
    diff = np.abs(y-pred)
    outlrs_indices = np.argwhere(diff> th_fix)
    inlrs_indices =  np.argwhere(diff<= th_fix)


    x_inlrs = [a[0] for a in x[inlrs_indices]]
    indices_inlrs = [a[0] for a in inlrs_indices]
    indices_outlrs = [a[0] for a in outlrs_indices]

    if debug: print "x_inlrs:",x_inlrs



    for k in outlrs_indices:
        k = k[0]
        
        if debug: print "looking for:",k,x[k]
        left_inlr = np.searchsorted(x_inlrs,x[k]) - 1
        right_inlr = left_inlr+1
        
        if left_inlr < 0:
            #let fix to the next 
            y[k] = y[indices_inlrs[0]]
        elif right_inlr < len(x_inlrs):
            xa = x_inlrs[left_inlr];
            xb = x_inlrs[right_inlr];
                
            ya = y[indices_inlrs[left_inlr]]
            yb = y[indices_inlrs[right_inlr]]

            if ya == yb:
                y[k] = ya
            else:
                #calculate regression
                y_pred = np.round(ya + (x[k]-xa)*float(yb-ya)/float(xb-xa))

                if debug: print left_inlr,x_inlrs[left_inlr],x_inlrs[right_inlr],y[indices_inlrs[left_inlr]],y[indices_inlrs[right_inlr]],y_pred
                y[k] = y_pred
        else:
            y[k] = y[indices_inlrs[left_inlr]]

        #insert prediction as in inliers
        x_inlrs = np.insert(x_inlrs,left_inlr+1,x[k])
        indices_inlrs = np.insert(indices_inlrs,left_inlr+1,k)
        #find inliers both size


    y = np.int32(np.round(y))
    return y

def fix_prediction(images_numbers, predictions):
    x = np.int32(images_numbers)
    y = np.int32(predictions)

    n = len(x)

    #repair obvious errors
    k = repair_1(x,y)

    #repair outliers
    y2 = detect_fix_outliers(x,y,th_detect=4,th_fix=2,debug=False)
    
    
    #repair obvious errors
    k = repair_1(x,y2)
    #repair latest errors
    k = repair_2(x,y2)

    return y2
