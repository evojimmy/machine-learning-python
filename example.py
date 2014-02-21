from pylab import *
from classifiers import *

def main():
    # First generate some test data
    class0_vectors = vstack((randn(50)*10+30, randn(50)*20+40)).T
    class1_vectors = vstack((randn(50)*20+40, randn(50)*30+30)).T
    class0_classes = zeros(50)
    class1_classes = ones(50)
    train_vectors = vstack((class0_vectors, class1_vectors))
    train_classes = hstack((class0_classes, class1_classes))

    # prepare plots
    f, axs = subplots(3, 1)

    # Plot all data
    ax = axs[0]
    ax.scatter(class0_vectors[:, 0], class0_vectors[:, 1], c='b', label='class0')
    ax.scatter(class1_vectors[:, 0], class1_vectors[:, 1], c='r', label='class1')
    xmin, xmax, ymin, ymax = ax.axis()

    # Multivariate Gaussian classifier
    model = MultivariateGaussian()
    model.train(train_vectors, train_classes)
    ax = axs[1]
    Xs = linspace(xmin, xmax, 50)
    Ys = linspace(ymin, ymax, 50)
    Cs = zeros((len(Xs), len(Ys)))
    for i, x in enumerate(Xs):
       for j, y in enumerate(Ys):
          Cs[i, j] = model.predict([(x, y)])[0]
    ax.pcolormesh(Xs, Ys, Cs.T)
    s1 = ax.scatter(class0_vectors[:, 0], class0_vectors[:, 1], c='b')
    s2 = ax.scatter(class1_vectors[:, 0], class1_vectors[:, 1], c='r')
    ax.legend((s1, s2), ('class0', 'class1'))
    ax.axis((xmin, xmax, ymin, ymax))

    # Nearest Neighbor classifier
    model = NearestNeighbor()
    model.train(train_vectors, train_classes)
    ax = axs[2]
    Xs = linspace(xmin, xmax, 50)
    Ys = linspace(ymin, ymax, 50)
    Cs = zeros((len(Xs), len(Ys)))
    for i, x in enumerate(Xs):
       for j, y in enumerate(Ys):
          Cs[i, j] = model.predict([(x, y)])[0]
    ax.pcolormesh(Xs, Ys, Cs.T)
    s1 = ax.scatter(class0_vectors[:, 0], class0_vectors[:, 1], c='b')
    s2 = ax.scatter(class1_vectors[:, 0], class1_vectors[:, 1], c='r')
    ax.legend((s1, s2), ('class0', 'class1'))
    ax.axis((xmin, xmax, ymin, ymax))

    # show plot
    show()

if __name__ == '__main__':
    main()
