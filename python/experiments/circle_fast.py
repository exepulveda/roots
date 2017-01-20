import numpy as np
import time

#see: https://en.wikipedia.org/wiki/Circumscribed_circle

def circle_vectorized(points):
	points_prima = points - points[0,:,:]
	points_prima_squared = np.sum(points_prima**2,axis=1)
	#print "points_prima.shape",points_prima.shape
	#print "points_prima_squared.shape",points_prima_squared.shape

	Dp = 2*(points_prima[1,0]*points_prima[2,1]-points_prima[1,1]*points_prima[2,0])
	#print "Dp.shape",Dp.shape

	#print "points_prima_squared",points_prima_squared,"Dp",Dp

	Up =np.empty((2,n))

	Up[0] = (points_prima[2,1]*points_prima_squared[1]-points_prima[1,1]*points_prima_squared[2])/Dp
	Up[1] = (points_prima[1,0]*points_prima_squared[2]-points_prima[2,0]*points_prima_squared[1])/Dp

	#print Up[:,:5]
	
	centers = Up + points[0,:,:]
	
	#distance of centers to first point
	diff = points[0,:,:] - centers
	radii = np.sqrt(np.sum(diff**2,axis=0))
	
	return centers, radii
	
def circle_loop(A,B,C):
	Ap = A-A
	Bp = B-A
	Cp = C-A

	B2=np.sum(Bp**2)
	C2=np.sum(Cp**2)

	#print "B2",B2,"C2",C2


	Dp = 2*(Bp[0]*Cp[1]-Bp[1]*Cp[0])

	#print "Dp",Dp

	Up =np.empty(2)

	Up[0] = (Cp[1]*(B2)-Bp[1]*(C2))/Dp

	Up[1] = (Bp[0]*(C2)-Cp[0]*(B2))/Dp


	U = Up + A
	
	#distance of centers to first point
	diff = A - U
	radii = np.sqrt(np.sum(diff**2,axis=0))
	return U, radii
	
if __name__ == "__main__":

	n = 100000

	points = np.random.random(size=(3,2,n)) * 100.0

	centers_1 = np.empty((2,n))
	radii_1 = np.empty(n)
	t1 = time.time()

	for i in range(n):
		A = points[0,:,i]
		B = points[1,:,i]
		C = points[2,:,i]
		centers_1[:,i],radii_1[i] =circle_loop(A,B,C)

	t2 = time.time()
	
	d1 = t2-t1

	print d1

	t1 = time.time()
	centers_2, radii_2 = circle_vectorized(points)
	t2 = time.time()
	
	d2 = t2-t1
	print d2
	
	print "improvement",d1/d2

	for i in range(n):
		assert centers_1[0,i] == centers_2[0,i]
		assert centers_1[1,i] == centers_2[1,i]
		assert radii_1[i] == radii_2[i]

	quit()
	A = np.array([3.0,4.0])
	B = np.array([2.0,-1.0])
	C = np.array([-1.0,5.0])

	t1 = time.time()
	for i in range(10000):


		Ap = A-A
		Bp = B-A
		Cp = C-A

		B2=np.sum(Bp**2)
		C2=np.sum(Cp**2)

		#print "B2",B2,"C2",C2


		Dp = 2*(Bp[0]*Cp[1]-Bp[1]*Cp[0])

		#print "Dp",Dp

		Up =np.empty(2)

		Up[0] = (Cp[1]*(B2)-Bp[1]*(C2))/Dp

		Up[1] = (Bp[0]*(C2)-Cp[0]*(B2))/Dp


		U = Up + A

		#print U

	t2 = time.time()
	print t2-t1
	print Up
		
	print 5.0/14.0, 27.0/14.0
