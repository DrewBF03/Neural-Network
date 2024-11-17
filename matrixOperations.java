////////////////////////////////////
// Name: Drew Ferrington         //
// CSC 475: Neural Networks     // 
// Created: Oct. 22, 2024      //
////////////////////////////////

/* ~ DREW'S DEV NOTES ~

Summary: matrixOperations provides mathematical operations for matrix/vector operations needed by MNISTNeuralNetwork; 
         compared to the previous implementation, additional operations like "subtract","elementWiseMultiply", and "transpose"
         help handle larger matrices, calculations for backpropagation, error calculations, and update gradients when training
         on MNIST dataset with its bigger input vectors and more complex structure.
         (NOTE: I reoriented sigmoidDerivative and costDerivative from previous implementation to be performed directly
         within backpropagate() using methods from matrixOperations to simplify repition/centralize backpropagation logic)
         */
        
        
        public class matrixOperations {
            
            /*1.) The "multiply(double[][] matrix, double[] vector)" method multiplies matrix by vector and returns resulting vector. 
            In MNISTNeuralNetwork, called by forward() and backpropagate(). During forward pass, computes weighted input to neurons
            by multiplying weights with activations from previous layer. In backpropagation, helps calculate error terms by propagating gradients backward through the network.*/
            //Calculate weighted sum of inputs for each neuron in a layer; Multiplies matrix by vector and returns resulting vector

            //Get weighted inputs during forward/back passes;Multiplies a two-dimensional matrix by a one-dimensional vector to compute weighted inputs for neurons during forward and backward passes.
            public static double[] multiply(double[][] matrix, double[] vector) {
                double[] result = new double[matrix.length]; //initialize the result vector with the same number of rows as the matrix
                for (int i = 0; i < matrix.length; i++) { //loop over each row 'i' of matrix
                double sum = 0;//initialize sum of dot product of row 'i'
                for (int j = 0; j < vector.length; j++) {//loop over each element 'j' in row and corresponding element in vector
                sum += matrix[i][j] * vector[j];//multiply matrix element at position [i][j] with vector element at [j] and add to sum
            }
            result[i] = sum;//passing computed sum to result vector at index 'i'
        }
        return result;
    }
    
    /*2.) The "add(double[] vector1, double[] vector2)" method performs element-wise addition of two vectors; 
    used in forward() in MNISTNeuralNetwork to add bias vector to weighted inputs before applying activation function.*/
    //Adds two vectors and returns resulting vector

    //Incorporate bias; Performs element-wise addition of two vectors to incorporate bias terms into neuron inputs before activation.
    public static double[] add(double[] vector1, double[] vector2) {
        double[] result = new double[vector1.length];//initialize result vector with same length as input vectors
    for (int i = 0; i < vector1.length; i++) {//loop over each element 'i' of vectors
    result[i] = vector1[i] + vector2[i];//add elements at index 'i' fro both vectors and store in result
}
return result;
}


/*3.) The "subtract(double[] vector1, double[] vector2)" method performs element-wise subtraction of one vector from another. 
In MNISTNeuralNetwork, it's utilized in backpropagate() to compute difference between the network's output and the target output, 
determining the error at the output layer.*/
//Subtracts one vector from another and returns resulting vector

//Error calculation; Subtracts the second vector from the first on an element-by-element basis to calculate errors between predicted and target outputs.
public static double[] subtract(double[] vector1, double[] vector2) {
    double[] result = new double[vector1.length]; //initialize result vector with same length as input vectors
    for (int i = 0; i < vector1.length; i++) {//loop over each element 'i' of vectors
    result[i] = vector1[i] - vector2[i];//subtract element of vector2 from vector1 at index 'i' and store in result
}
return result;
}


/*4.) The "subtract(double scalar, double[] vector)" method subtracts each element of a vector from a scalar value;
used in backpropagate() to calculate (1 - output) during computation of derivative of sigmoid activation function.*/
//Subtracts each element of a vector from a scalar and returns resulting vector

//Subtracts each element of a vector from a scalar value to compute derivatives of the sigmoid activation function during backpropagation.
public static double[] subtract(double scalar, double[] vector) {
    double[] result = new double[vector.length];//initialize result vector with same length as input vector
    for (int i = 0; i < vector.length; i++) {//loop over each element 'i' of vector
    result[i] = scalar - vector[i];//subtract vector element at index 'i' from scalar and store result
}
return result;
}

/*5.) The "sigmoid(double[] vector)" method applies sigmoid activation function element-wise to a vector; called by forward() in MNISTNeuralNetwork 
to transform weighted inputs into activations between 0 and 1, introducing non-linearity to the model.*/
//Applies sigmoid activation function to each element of a vector, returning resulting vector

//Applies the sigmoid activation function to each element of a vector to transform weighted inputs into activations between 0 and 1, introducing non-linearity to neuron activations for complex pattern recognition.
public static double[] sigmoid(double[] vector) {
    double[] result = new double[vector.length]; //initialize result vector with same length as input vector
    for (int i = 0; i < vector.length; i++) {//loop over each element 'i' of vector
    result[i] = 1.0 / (1.0 + Math.exp(-vector[i]));//apply sigmoid function to element at index 'i' and store in result
}
return result;
}

/*6.) The "elementWiseMultiply(double[] vector1, double[] vector2)" method performs element-wise multiplication of two vectors; used in backpropagate() 
to multiply error terms with derivative of activation function during gradient computation.*/
//Element-wise multiplication of two vectors

//Gradient calculations in backpropagate(); Performs element-wise multiplication of two vectors to apply activation function derivatives to errors during gradient calculations.
public static double[] elementWiseMultiply(double[] vector1, double[] vector2) {
    double[] result = new double[vector1.length]; //initialize result vector with same length as input vectors
    for (int i = 0; i < vector1.length; i++) {//loop over each element 'i' of vectors
    result[i] = vector1[i] * vector2[i];//multiply elements at index 'i' from both vectors and store in result
}
return result;
}

/*7.) The "transpose(double[][] matrix)" method transposes a given matrix, swapping its rows and columns. In MNISTNeuralNetwork, used in backpropagate() 
to transpose weight matrices, aligning dimensions for matrix multiplication when propagating errors backward.*/
//Returns transposed matrix

//Transposes matrix by swapping its rows with columns to align dimensions for accurate matrix multiplication for error propagation in backpropagation.
public static double[][] transpose(double[][] matrix) {
    double[][] result = new double[matrix[0].length][matrix.length];//initialize result matrix with dimensions swapped (rows become columns and vice versa)
    for (int i = 0; i < matrix.length; i++) {//loop over each row 'i' of input matrix
    for (int j = 0; j < matrix[0].length; j++) {//loop over each column 'j' of input matrix
    result[j][i] = matrix[i][j];//assign element at position [i][j] in input matrix to position [j][i] in result matrix
}
        }
        return result;
    }
}
