import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as img
from PIL import Image


def construct_inputs(A, img1, img2, display=False):
    image1 = img.imread(img1)
    image2 = img.imread(img2)

    data1 = np.asarray(image1)
    data2 = np.asarray(image2)

    a = A[0][0]
    b = A[0][1]
    c = A[1][0]
    d = A[1][1]

    x1 = data1*a + data2*b
    x2 = c*data1 + d*data2

    if display:
        plt.subplot(221)
        plt.imshow(data1)
        plt.subplot(222)
        plt.imshow(data2)
        plt.subplot(223)
        plt.imshow(x1.astype(int)) 
        plt.subplot(224)
        plt.imshow(x2.astype(int))
        plt.show()

    return x1, x2

if __name__ == "__main__":
    A = [[1/4, 3/4], [2/3, 1/3]]
    x1, x2 = construct_inputs(A, 'fig1.jpg', 'fig2.jpg')
    m, n, R = x1.shape
    x1 = np.reshape(x1/255, (m*n, R))
    x1 = x1 - np.mean(x1)
    x2 = np.reshape(x2/255, (m*n, R))
    x2 = x2 - np.mean(x2)

    # Step1: Rotate out Principal Component Direction
    theta0 = 0.5*np.arctan(-2*(sum(x1*x2))/sum(x1**2 -x2**2))
    Us = np.array([[np.cos(theta0)[0], np.sin(theta0)[0]],
                  [-np.sin(theta0)[0], np.cos(theta0)[0]]])

    # Step2: Undo scaling of singular values
    sig1 = sum((x1*np.cos(theta0) + x2*np.sin(theta0))**2)[0]
    sig2 = sum((x1*np.cos(theta0-np.pi/2) + x2*np.sin(theta0-np.pi/2))**2)[0]
    Sigma = np.array([[1/np.sqrt(sig1), 0], [0, 1/np.sqrt(sig2)]])

    # Step3: Make probability density separable
    x1bar = Sigma[0][0]*(Us[0][0]*x1 + Us[0][1]*x2)
    x2bar = Sigma[1][1]*(Us[1][0]*x1 + Us[1][1]*x2)
    phi0 = 0.25*np.arctan(-sum(2*(x1bar**3)*x2bar-2*(x2bar**3)*x1bar)/
            sum(3*(x1bar**2)*(x2bar**2)-0.5*(x1bar)**4 -0.5*(x2bar)**4))[0]
    V = np.array([[np.cos(phi0), np.sin(phi0)], [-np.sin(phi0), np.cos(phi0)]])

    A1 = Us[0][0]*x1 + Us[1][0]*x2
    A2 = Us[0][1]*x1 + Us[1][1]*x2

    B1 = Sigma[0][0]*A1 + Sigma[1][0]*A2
    B2 = Sigma[0][1]*A1 + Sigma[1][1]*A2

    S1 = V[0][0]*B1 + V[1][0]*B2
    S2 = V[0][1]*B1 + V[1][1]*B2

    S1 = 500000* S1.reshape((m,n,3))
    plt.imshow((S1).astype(int))
    plt.show()

    S2 = 500000* S2.reshape((m,n,3))
    plt.imshow((S2).astype(int))
    plt.show()
