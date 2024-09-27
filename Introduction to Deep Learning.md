# Introduction to Deep Learning

## Why Deep Learning?

- Hand-engineered features are time-consuming, brittle, and not scalable in practice.
- **Question:** Can we learn the underlying features directly from data?

## Why Now?

1. **Big Data**: We now have access to vast amounts of data.
2. **GPUs**: The advent of GPUs allows for massive parallelization, accelerating computations.
3. **Software**: Frameworks like TensorFlow simplify the creation of machine learning models that can run in various environments.

## Neuron (Perceptron)

A neuron performs a series of operations on inputs:

1. A neuron receives inputs \(x_1, x_2, ..., x_n\), each multiplied by a corresponding weight \(w_1, w_2, ..., w_n\), forming the sum:
   \[
   \text{output} = x_1 \times w_1 + x_2 \times w_2 + \ldots + x_n \times w_n
   \]
2. This sum is then passed through a **non-linear activation function**.
3. A **bias** is added before the activation function. This bias value allows shifting the activation function, which can be critical for successful learning.

## Activation Functions

### Sigmoid

The sigmoid function is commonly used in neural networks, especially for binary classification. It maps input values to a probability range between 0 and 1, making it useful for calculating probabilities. The sigmoid function is defined as:

\[
\sigma(x) = \frac{1}{1 + e^{-x}}
\]

### ReLU (Rectified Linear Unit)

ReLU returns the input if it's positive, and 0 if it's negative. It is commonly used in deep learning due to its efficiency:

\[
f(x) = \text{max}(0, x)
\]

## Why Do We Need Activation Functions?

Activation functions introduce **non-linearities** into the network, enabling the model to capture complex patterns in the data.

![alt text](image.png)

![alt text](image-1.png)

## Building a Dense Layer

A dense layer is fully connected, meaning each input neuron is connected to each output neuron. Here's a graphical representation:

![Dense Layer](image-2.png)

## Single Layer Neural Network

A single layer neural network consists of a set of inputs passing through a hidden layer to produce an output:

![Single Layer Network](image-3.png)

## Deep Neural Network

In a deep neural network, the input passes through multiple hidden layers before producing the output. This allows the network to learn more complex features:

![Deep Neural Network](image-4.png)

## Loss Functions

Once the network is built, it needs to be trained. The training process involves minimizing the **loss** between the predicted output and the actual output. The closer the predicted output is to the actual value, the lower the loss and the more accurate the model.

### Types of Loss Functions

- **Empirical Loss**: Measures the difference between actual and predicted values across the dataset.
- **Mean Squared Error (MSE)**: Commonly used for regression tasks. Instead of a binary output, we want to predict a real number, for ex, grade for students in a class
- **Cross-Entropy Loss**: Commonly used for classification tasks.
- **Binary Cross Entropy Loss**: can be used with models that output a probability between 0 and 1
- **Softmax Cross Entropy with Logits**: Softmax Cross Entropy with Logits is a widely used loss function in neural networks for multi-class classification problems. It combines two essential components:
  1.  **_Softmax_**: a normalization function that converts unnormalized log probabilities (logits) into a probability distribution over multiple classes.
  2.  **_Cross Entropy_**: a measure of the difference between the predicted probabilities and the true labels.
  3.  **_Logits_** are the unnormalized output values from a neural network’s final layer, typically a linear or affine transformation of the input data. They represent the “scores” or “energies” assigned to each class.

### Gradient Descent Optimization

we have the loss function which will tell us the loss and our objective is to find the network weights that achieve the lowest loss which means we are trying to find the weights for the neural network based on all the data that we have seen.

Weight is nothing but just a vector which we are trying to find and our loss is a function of the network weights

**Step 1**: If we have only two input, we can plot our loss function in the 3d graph

**Step 2**: Pick a random point on the graph, from that place lets computer the gradient (local estima of where is slop is increasings)

**Step 3**: Take small step in opposite direction of gradient by updating weights

**Step 4**: Repeat step 2 & 3 until we found a local minimum

![alt text](image-5.png)

Gradient tells us which way is up in that landscape and it also tells how our loss is changing as a function of all our weightsm how does it do that? To minimize the loss, we use gradient descent to update the weights in the network. There are three key types of gradient descent:

1. **Batch Gradient Descent**: Computes the gradient for the entire dataset.
2. **Stochastic Gradient Descent (SGD)**: Computes the gradient for a single training example.
3. **Mini-Batch Gradient Descent**: Computes the gradient for small batches of training data, combining the advantages of both batch and stochastic methods.

#### Backpropogation

Consider a single input going to a single neuron and that neuron output is going as an input to another neuron

![alt text](image-6.png)

How does a small change in one weight affect the loss (whether its going up or down)? Refer image in PDF for the formula

#### Learning Rate

- A **small learning rate** converges slowly and risks getting stuck in local minima.
- A **large learning rate** can cause the model to overshoot and become unstable.
- **Stable learning rates** ensure smooth convergence and avoid local minima.

**Solution**:

1. Experiment with different learning rates.
2. Use adaptive learning rates like **Adam**, which adjust according to the landscape.

## Batched Gradient Descent

SGD computes the gradient for a mini-batch (e.g., 32 examples) instead of the entire dataset, making it more efficient and scalable. Mini-batches lead to fast training and allow parallelization, especially on GPUs.

## Regularization: Dropout and Early Stopping

### Underfitting vs. Overfitting

- **Underfitting**: Occurs when the model is too simple and fails to capture patterns in the training data.
- **Overfitting**: Happens when the model is too complex and learns noise from the training data, leading to poor generalization.

### Regularization Techniques

To prevent overfitting, we use regularization techniques such as:

1. **Dropout**: Randomly dropping neurons during training forces the network to learn more robust features by preventing it from relying on specific neurons.
2. **Early Stopping**: Stops training when validation error stops improving, preventing overfitting.
