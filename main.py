# Hypersphere sampling

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import chisquare

def sample_hypersphere(dims=1, samples=1, radius=1, x_base=None):
    """
    Sample the hull of a hypersphere of dimension dims with radius radius around the origin x_base
    """
    if x_base is None:
        x_base=np.zeros(dims)
    points=np.random.randn(samples, dims)
    return radius * points / np.linalg.norm(points, axis=1)[:, None]+x_base




# #Test 2D
# points=sample_hypersphere(2, 1000, 0.2, np.array([0.5, 0.5]))
# plt.scatter(points[:, 0], points[:, 1])
# plt.show()
#
# #Test 3D
# points=sample_hypersphere(3, 1000, 2.6, np.array([0.5, 2.5, 7.5]))
# fig=plt.figure()
# ax=fig.add_subplot(111, projection='3d')
# ax.scatter(points[:, 0], points[:, 1], points[:, 2])
# plt.show()

# Evaluation function of model with parameters and test if result is not optimal anymore
def evaluate(points, f, f_optimal, base_parameter=np.nan, scale_parameter=np.nan):
    """
    Evaluate the function f which is our model simulationon the points
    f_optimal should return in our case 0 or 1 -> It's an indicator function
    At some point the parameters have to be scaled to the original values instead of the [0,1] range and the
    """
    if np.isnan(base_parameter).any():
        base_parameter=np.zeros(len(points[0]))
    if np.isnan(scale_parameter).any():
        scale_parameter=np.ones(len(points[0]))

    # scale points to original values
    points = points * scale_parameter + base_parameter  # transforms z-scored points to original parameters
    # apply f to every point in points
    result= [f(point) for point in points]
    # test of result is not optimal anymore (would be FOC1=1, FOC2=0 ...)
    result = [f_optimal(point) for point in result]
    return np.mean(result)

def f_optimal(point):
    """
    Indicator function for optimal solution
    I want to test if my coordinates are in the hypercube [0, 1]^4
    """
    return np.all(point >= 0) and np.all(point <= 1)

def f(point):
    """
    My model simulation
    Example: I want to do nothing with the point to don't make it too complicated
    """
    return point# * np.array([0.5, 0.5, 0.5, 0.5])


# # Test evaluate function
#points=sample_hypersphere(4, 1000, 1)
#print(evaluate(points, f, f_optimal, np.array([0.5, 2.5, 1, 0.5]), np.array([0.5, 2.5, 1, 0.5])))
#print(evaluate(points, f, f_optimal, np.array([0.5, 2.5, 1, 0.5]), np.array([0.5, 2.5, 1, 0.5])) == 1)


if __name__ == '__main__':
    # I want to test how many points lie in the hypercube [0, 1]^4 with inreasing radius of the hypersphere
    SampleSize = 1000
    Dimensions = 4
    radius = np.linspace(0.0, 2.0, 100) # times the scale parameter it results in a hypercube with a dimater of 2
    base_parameter = np.array([0.5, 0.5, 0.5, 0.5]) #the middle of the hypercube

    optimal=np.zeros(len(radius))
    for i in range(len(radius)):
        points=sample_hypersphere(Dimensions, SampleSize, radius[i]) # sample points around origin, don't apply base_parameter here
        # they would be also scaled with the scale_parameter afterwards
        optimal[i]=evaluate(points, f, f_optimal, base_parameter)

    plt.plot(radius, optimal*100)
    plt.xlabel('Radius of hypersphere')
    plt.ylabel('Optimal points [%]')
    plt.show()
    # cureve should start to decrease at a radius of 0.5 because it then starts to leave the hypercube
    # as the origin is in the middle of the hypercube



    # test the hypersphere sampling, if it is really uniform
    points=sample_hypersphere(2, 200000, 1)
    # divide the circle in 100 sectors and count how many points are in each sector
    # if it is uniform, each sector should have 10 points
    # if it is not uniform, some sectors should have more points than others

    # calculate the angle of each point
    plt.subplot(121)
    plt.scatter(points[:, 0], points[:, 1], s=0.1)
    plt.subplot(122)
    angles=np.arctan2(points[:, 1], points[:, 0])
    # transform angles to [0, 2pi]
    angles[angles<0]=angles[angles<0]+2*np.pi
    # divide the circle in 100 sectors
    sectors=np.linspace(0, 2*np.pi, 100)
    # count how many points are in each sector
    counts=plt.hist(angles, bins=sectors)
    plt.show()
    # test if the counts are uniform with a chi-squared test

    print(chisquare(counts[0]))

    # test the hypersphere sampling, if it is really uniform
    # in 3D
    points=sample_hypersphere(3, 200000, 1)
    # divide the sphere in N equal sectors and count how many points are in each sector

    # calculate the angle of each point
    fig=plt.figure()
    ax=fig.add_subplot(131, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.1)

    phi=np.arctan2(points[:, 1], points[:, 0])
    theta=np.arctan2(np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2), points[:, 2])
    # transform angles to [0, 2pi]
    phi[phi < 0]= phi[phi < 0] + 2 * np.pi
    theta[theta < 0]= theta[theta < 0] + 2 * np.pi
    # divide the circle in 100 sectors
    sectors_phi=np.linspace(0, 2 * np.pi, 100)
    sectors_theta=np.linspace(0, np.pi, 100)
    # count how many points are in each sector
    ax=fig.add_subplot(132)
    counts_phi=plt.hist(phi, bins=sectors_phi)
    ax=fig.add_subplot(133)
    counts_theta=plt.hist(theta, bins=sectors_theta)



    # test if the counts are uniform with a chi-squared test

    print(chisquare(counts_phi[0]))
    print(chisquare(counts_theta[0]))

    # calculate the area integral of the hypersphere for phi
    A = lambda theta: -2*np.pi*((np.cos(theta[1]))-(np.cos(theta[0])))


    area=np.zeros(len(counts_theta[0]))
    for i in range(len(counts_theta[0])):
        area[i]=A(sectors_theta[i:i + 2])
    # calculate the expected counts
    # normalize the counts by the area of the sector
    counts2_theta=counts_theta[0]/area
    #plot the normalized counts bar plot
    plt.bar(sectors_theta[:-1], counts2_theta, width=sectors_theta[1]-sectors_theta[0])
    plt.show()
    # test if the counts are uniform with a chi-squared test
    # I think there is a bug which I can not find right now
    print(chisquare(np.round(counts2_theta).astype(int)))









