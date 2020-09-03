import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x)) >= 0:
            return 1.0
        return -1.0

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        batch_size = 1
        while True:
            mistakes = 0
            for x, y in dataset.iterate_once(batch_size):
                y_hat = self.get_prediction(x)
                if y_hat == nn.as_scalar(y):
                    continue
                else:
                    mistakes += 1
                    self.w.update(x, -1 * y_hat)
            if mistakes == 0:
                return


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.c1 = nn.Parameter(1, 20)
        self.b1 = nn.Parameter(1, 20)
        self.c2 = nn.Parameter(20, 1)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        f1 = nn.AddBias(nn.Linear(x, self.c1), self.b1)
        layer1 = nn.ReLU(f1)
        f2 = nn.AddBias(nn.Linear(layer1, self.c2), self.b2)
        return f2

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_hat = self.run(x)
        return nn.SquareLoss(y_hat, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = dataset.x.shape[0]
        while True:
            total_loss = 0
            num_samples = 0
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                #print(nn.as_scalar(loss))
                total_loss += nn.as_scalar(loss)
                c1, c2, b1, b2 = nn.gradients(loss, [self.c1, self.c2, self.b1, self.b2])
                self.c1.update(c1, -.01)
                self.c2.update(c2, -.01)
                self.b1.update(b1, -.01)
                self.b2.update(b2, -.01)
                num_samples += 1
            if total_loss / num_samples <= .01:
                return


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.c1 = nn.Parameter(784, 300)
        self.b1 = nn.Parameter(1, 300)

        self.c2 = nn.Parameter(300, 150)
        self.b2 = nn.Parameter(1, 150)

        # self.c3 = nn.Parameter(50, 10)
        # self.b3 = nn.Parameter(1, 10)
        #
        # self.c4 = nn.Parameter(10, 10)
        # self.b4 = nn.Parameter(1, 10)

        self.c3 = nn.Parameter(150, 50)
        self.b3 = nn.Parameter(1, 50)

        self.c4 = nn.Parameter(50, 10)
        self.b4 = nn.Parameter(1, 10)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        f1 = nn.AddBias(nn.Linear(x, self.c1), self.b1)
        layer1 = nn.ReLU(f1)
        f2 = nn.AddBias(nn.Linear(layer1, self.c2), self.b2)
        layer2 = nn.ReLU(f2)
        f3 = nn.AddBias(nn.Linear(layer2, self.c3), self.b3)
        layer3 = nn.ReLU(f3)
        f4 = nn.AddBias(nn.Linear(layer3, self.c4), self.b4)
        return f4

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_hat = self.run(x)
        return nn.SoftmaxLoss(y_hat, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = dataset.x.shape[0]
        batch_size //= 5
        while True:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                # alpha = .3 - nn.as_scalar(loss)
                #print(nn.as_scalar(loss))
                c1, c2, c3, c4, b1, b2, b3, b4 = nn.gradients(loss, [self.c1, self.c2, self.c3, self.c4, self.b1, self.b2, self.b3, self.b4])
                self.c1.update(c1, -.5)
                self.c2.update(c2, -.3)
                self.c3.update(c3, -.1)
                self.c4.update(c4, -.05)
                self.b1.update(b1, -.5)
                self.b2.update(b2, -.3)
                self.b3.update(b3, -.1)
                self.b4.update(b4, -.05)
                # self.c1.update(c1, -alpha)
                # self.c2.update(c2, -alpha)
                # self.c3.update(c3, -alpha)
                # self.c4.update(c4, -alpha)
                # self.b1.update(b1, -alpha)
                # self.b2.update(b2, -alpha)
                # self.b3.update(b3, -alpha)
                # self.b4.update(b4, -alpha)
            if dataset.get_validation_accuracy() >= .975:
                return



class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.d = 400
        self.w = nn.Parameter(self.num_chars, self.d)
        self.w_hidden = nn.Parameter(self.d, self.d)

        self.prescore = nn.Parameter(self.d, self.d)
        self.score = nn.Parameter(self.d, 5)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        # "*** YOUR CODE HERE ***"
        z = nn.Linear(xs[0], self.w)
        h = z
        for c in xs[1:]:
            h = nn.Add(nn.Linear(c, self.w), nn.Linear(z, self.w_hidden))
            z = nn.ReLU(h)
        return nn.Linear(nn.Linear(h, self.prescore), self.score)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_hat = self.run(xs)
        return nn.SoftmaxLoss(y_hat, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # batch_size = dataset.x.shape[0]
        batch_size = 250
        # batch_size //= 5
        while True:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                #print(nn.as_scalar(loss))
                w, w_hidden, score = nn.gradients(loss, [self.w, self.w_hidden, self.score])
                self.w.update(w, -.01)
                self.w_hidden.update(w_hidden, -.01)
                self.score.update(score, -.08)

            if dataset.get_validation_accuracy() >= .88:
                return
