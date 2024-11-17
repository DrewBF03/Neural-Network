////////////////////////////////////
// Name: Drew Ferrington         //
// CSC 475: Neural Networks     // 
// Created: Oct. 22, 2024      //
////////////////////////////////

/* ~ DREW'S DEV NOTES ~

Summary: MNISTNeuralNetwork implementation is designed to recognize handwritten digits using MNIST dataset. Compared to smallNeuralNetwork implementation, this version handles larger input vectors (28x28 pixels images flattened into 784-element vectors) and 
         a more complex network structure. Key enhancements include additional methods like `loadData`, `trainNetwork`, `backpropagate`, and user interface functions. These additions enable the network to manage larger datasets, perform complex calculations for backpropagation, 
         calculate errors, and update gradients efficiently when training on the MNIST dataset with its bigger input vectors and more sophisticated architecture.
         (NOTE: "sigmoidDerivative() and "costDerivative() from the previous implementation have been reoriented to be performed directly within "backpropagate()" using methods from matrixOperations 
         This simplification reduces repetition and centralizes the backpropagation logic.)

         Functionality:
         
         9.) The UI methods ("showMenu()", "runNetworkOnTestingData()", "displayMisclassifiedTestingImages()") provide command-line interface allowing user to train network,
         evaluate performance, save or load the network state, and visualize data.
         
         12.) The "saveNetworkState()" and "loadNetwork()" methods save and load the network's weights and biases to and from a file
         
         (NOTE: MNISTNeuralNetwork uses matrixOperations for mathematical operations required in forward and backward propagation. Methods like 
         "multiply()", "add()", "subtract()", "elementWiseMultiply()", "sigmoid()", and "transpose()" handle larger matrices and complex calculations, enabling efficient training on the MNIST dataset.)
         */
        
        import java.io.*;//input/output operations
        import java.util.*;//List,ArrayList,Scanner,etc
        
        public class MNISTNeuralNetwork {
            
            //Network parameters
            private final int inputSize = 784; //# of input (28x28 pixels = 784 input nodes)
            private final int hiddenSize = 15; //15 nodes in hidden layer
            private final int outputSize = 10; //10 nodes for output (digits 0–9)
            
            //Network weights and biases
            private double[][] W1; //weights from input to hidden layer
            private double[][] W2; //weights from hidden to output layer
            private double[] B1; //biases for hidden layer
            private double[] B2; //biases for output layer
            
            //Activations for layers
            private double[] A1; //activations for hidden layer
            private double[] A2; //activations for output layer
            
            private final double eta = 3.0; //learning rate (gradient descent)
            
            //Training data and labels
            private List<double[]> trainingData = new ArrayList<>();//list to store training input data
            private List<double[]> expectedOutput = new ArrayList<>();//list to store training labels (one-hot encoded)
            
            //Scanner for user input (command-line interface)
    private Scanner scanner;
    
    
    /*1.) The constructor ("MNISTNeuralNetwork()") initializes neural network with random weights/biases, 
    setting up network parameters (input size, hidden layer size, output size); contrasting
    smallNeuralNetwork, it prepares the network to handle the larger MNIST dataset.*/

    //Constructor (initializes neural network with random weights and biases)
    public MNISTNeuralNetwork() {
        W1 = initializeWeights(hiddenSize, inputSize);
        W2 = initializeWeights(outputSize, hiddenSize);
        B1 = initializeBiases(hiddenSize);
        B2 = initializeBiases(outputSize);
        
        scanner = new Scanner(System.in);//initialize scanner for user input
    }
    
    //main
    public static void main(String[] args) {
        MNISTNeuralNetwork network = new MNISTNeuralNetwork();
        network.loadData("mnist_train.csv", network.trainingData, network.expectedOutput);//load training data/labels(CSV file)
        
        // Load testing data from CSV file for evaluation
        List<double[]> testingData = new ArrayList<>();//list to store testing input data
        List<double[]> testingLabels = new ArrayList<>();//list to store testing labels
        network.loadData("mnist_test.csv", testingData, testingLabels);//load testing data and labels
        network.showMenu(testingData, testingLabels);//start interactive menu
    }
    
    
    /*2.) The "loadData(String fileName, List<double[]> data, List<double[]> labels)" method loads data from 
    CSV file into provided lists for data and labels; extends functionality by supporting large datasets 
    through file I/O operations, unlike static data approach in smallNeuralNetwork.*/

    //Loads data from a CSV file into the provided data and labels lists.
    public void loadData(String fileName, List<double[]> data, List<double[]> labels) {
        try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {//open file for reading
            String line;//variable to hold each line read from file
            // Read each line representing a data sample
            while ((line = reader.readLine()) != null) {//read until end of file
                String[] values = line.split(",");//split line by commas to get individual values
                double[] input = new double[inputSize];//array to hold input pixel values
                double[] output = new double[outputSize];//array to hold one-hot encoded label
                
                // Normalize pixel values (0-255) to (0-1)
                for (int i = 1; i < values.length; i++) {//start from index 1 as index 0 is label
                    input[i - 1] = Double.parseDouble(values[i]) / 255.0;//normalize and store pixel value
                }
                
                // One-hot encode the label (0-9)
                int label = Integer.parseInt(values[0]);//parse label from first value in line
                output[label] = 1.0;//set the corresponding index in the output array to 1.0
                
                // Add input and output arrays to their respective lists
                data.add(input);//add the input data to the data list
                labels.add(output);//add the one-hot encoded label to the labels list
            }
            System.out.println("Data loaded from " + fileName);//print confirmation message
        } catch (IOException e) {//handle exceptions
            System.err.println("Error reading file: " + e.getMessage());//print error
        }
    }
    
    
    /*3.) The "trainNetwork(int epochs, int minibatchSize)" method trains neural network using minibatch gradient descent; scales up the training process from 
    previous implementation by handling large datasets and implementing stochasticity through shuffling.*/

    //Trains the neural network by iterating over a specified number of epochs, shuffling the training data for each epoch to ensure randomness, dividing the data into minibatches of a defined size, computing gradients through backpropagation for each minibatch, and updating the network's weights and biases based on the accumulated gradients to minimize the loss function.

    //Trains the neural network using minibatch gradient descent.
    public void trainNetwork(int epochs, int minibatchSize) {
        for (int epoch = 0; epoch < epochs; epoch++) {//loop over # of epochs (complete pass through training dataset)
            System.out.println("\nEpoch " + (epoch + 1) + " Results:");//print epoch # to track training
            
            // Shuffle training data indices for stochasticity
            List<Integer> indices = new ArrayList<>();// List to store indices of training samples for shuffling
            for (int i = 0; i < trainingData.size(); i++) {
                indices.add(i);// Add each index to the list
            }
            Collections.shuffle(indices);// Shuffle the indices to randomize the training samples and prevent training bias; Collections=utility class from java.util for manipulating lists/sets/etc, like sort, search, or SHUFFLE
            
            // Process minibatches
            for (int i = 0; i < trainingData.size(); i += minibatchSize) { // Loop over training data in steps of minibatchSize to create minibatches
                // Initialize accumulators for gradients of weights/biases; will store sum of gradients for current minibatch
                double[][] W1GradientSum = new double[hiddenSize][inputSize];//accumulator for W1 gradients
                double[][] W2GradientSum = new double[outputSize][hiddenSize];//accumulator for W2 gradients
                double[] B1GradientSum = new double[hiddenSize];//accumulator for B1 gradients
                double[] B2GradientSum = new double[outputSize];//accumulator for B2 gradients
                
                // iterate over each sample in current minibatch without exceeding dataset size to process for gradient calculations
                for (int j = i; j < i + minibatchSize && j < trainingData.size(); j++) {
                    int idx = indices.get(j);//retrieve shuffled index for current sample
                    double[] input = trainingData.get(idx);//get input vector (pixel values) for current training sample
                    double[] target = expectedOutput.get(idx);//get target output vector(one-hot encoded label) for current training sample
                    
                    // Perform backpropagation to get gradients on current input and target; forward pass followed by calculating errors and gradients
                    GradientValues gradients = backpropagate(input, target);
                    
                    // Accumulate gradients from current sample into minibatch gradient sums to prepare them for averaging/updating network parameters
                    accumulateGradients(gradients, W1GradientSum, W2GradientSum, B1GradientSum, B2GradientSum);
                }
                
                // After processing minibatch, update weights and biases using accumulated gradients; gradients scaled by minibatch size to compute average gradient
                updateWeightsAndBiasesFromGradients(W1GradientSum, W2GradientSum, B1GradientSum, B2GradientSum, minibatchSize);
            }
            
            //after completing all minibatches in current epoch, display training statistics
            displayEpochStatistics(epoch + 1, trainingData, expectedOutput);
        }
    }
    
    
    //Displays the menu for user interaction.
    public void showMenu(List<double[]> testingData, List<double[]> testingLabels) {
        boolean networkTrained = false; // Flag to check if the network has been trained or loaded
        
        try {
            while (true) {
                System.out.println("\n--- MNIST Neural Network Menu ---");
                System.out.println("1. Train the network");
                System.out.println("2. Load a pre-trained network");
                System.out.println("3. Display network accuracy on training data");
                System.out.println("4. Display network accuracy on testing data");
                System.out.println("5. Run network on testing data showing images and labels");
                System.out.println("6. Display the misclassified testing images");
                System.out.println("7. Save the network state to file");
                System.out.println("0. Exit");
                System.out.print("Enter your choice: ");
                int choice;
                try {
                    choice = scanner.nextInt();
                } catch (InputMismatchException e) {
                    System.out.println("Invalid input. Please enter a number.");
                    scanner.nextLine(); // Clear the invalid input
                    continue;
                }
                switch (choice) {
                    case 1:
                    // Train the network
                    trainNetwork(30, 10); // 30 epochs, minibatch size of 10
                    networkTrained = true;
                    break;
                    case 2:
                    // Load a pre-trained network
                    loadNetwork();
                    networkTrained = true;
                    break;
                    case 3:
                    // Display accuracy on training data
                    if (networkTrained) {
                        displayEpochStatistics(-1, trainingData, expectedOutput);
                    } else {
                        System.out.println("Please train or load the network first.");
                    }
                    break;
                    case 4:
                    // Display accuracy on testing data
                    if (networkTrained) {
                        displayEpochStatistics(-1, testingData, testingLabels);
                    } else {
                        System.out.println("Please train or load the network first.");
                    }
                    break;
                    case 5:
                    // Run network on testing data showing images and labels
                    if (networkTrained) {
                        runNetworkOnTestingData(testingData, testingLabels);
                    } else {
                        System.out.println("Please train or load the network first.");
                    }
                    break;
                    case 6:
                    // Display misclassified testing images
                    if (networkTrained) {
                        displayMisclassifiedTestingImages(testingData, testingLabels);
                    } else {
                        System.out.println("Please train or load the network first.");
                    }
                    break;
                    case 7:
                    // Save the network state to file
                    if (networkTrained) {
                        saveNetworkState();
                    } else {
                        System.out.println("Please train or load the network first.");
                    }
                    break;
                    case 0:
                    // Exit the program
                    System.out.println("Exiting...");
                    return;
                    default:
                    System.out.println("Invalid option. Try again.");
                }
                }
            } finally {
                if (scanner != null) {
                    scanner.close(); // Close the Scanner when exiting the program
                }
            }
        }

        //Initializes weights with random values between -1 and 1.
        private double[][] initializeWeights(int rows, int cols) {
            double[][] weights = new double[rows][cols];   // Create a matrix for weights
            Random rand = new Random();                    // Create a Random object for generating random numbers
            
            // Randomly initialize weights
            for (int i = 0; i < rows; i++) {               // Loop over rows
                for (int j = 0; j < cols; j++) {           // Loop over columns
                    weights[i][j] = 2 * rand.nextDouble() - 1;   // Assign random value between -1 and 1
                }
            }
            return weights;    // Return the initialized weight matrix
        }
    
        //Initializes biases with random values between -1 and 1.
        private double[] initializeBiases(int size) {
            double[] biases = new double[size];    // Create an array for biases
            Random rand = new Random();            // Create a Random object for generating random numbers
            
            // Randomly initialize biases
            for (int i = 0; i < size; i++) {       // Loop over the size of biases
                biases[i] = 2 * rand.nextDouble() - 1;   // Assign random value between -1 and 1
            }
            return biases;     // Return the initialized bias vector
        }
    
        
        /*4.) The "forward(double[] input)" method performs forward pass through the network; similar to smallNeuralNetwork but 
        scaled to handle larger input vectors; computes weighted inputs and applies sigmoid activation function to introduce non-linearity.*/

        //Processes input through hidden and output layers by computing weighted sums, adding biases, and applying sigmoid activation to generate predictions.

        //Performs a forward pass through the network.
        private double[] forward(double[] input) {
        // Compute activations for hidden layer
        A1 = matrixOperations.sigmoid(matrixOperations.add(matrixOperations.multiply(W1, input), B1));
        // Compute activations for output layer
        A2 = matrixOperations.sigmoid(matrixOperations.add(matrixOperations.multiply(W2, A1), B2));
        return A2; // Return the final output activations
    }
    
    /*5.) The "backpropagate(double[] input, double[] target)" method performs backpropagation and computes gradients; integrates derivative 
    calculations in this method using existing methods from matrixOperations, simplifying the code/centralizing backpropagation logic; calculates error 
    terms and propagates gradients backward through the network.*/

    //Performs forward pass, computes output errors, calculates delta values using sigmoid derivatives, propagates errors to hidden layer, and computes gradients for parameters.

    // Performs backpropagation to compute gradients based on input and target output.
public GradientValues backpropagate(double[] input, double[] target) {
    
    // Forward pass to get output activations
    double[] output = forward(input);    // Perform forward pass to compute output activations
    
    // Compute output layer error delta2
    double[] delta2 = matrixOperations.elementWiseMultiply(      // Element-wise multiplication for delta2
        matrixOperations.subtract(output, target),                 // Compute (output - target), the cost derivative
        matrixOperations.elementWiseMultiply(                      // Multiply by sigmoid derivative
            output,                                                // Current output activations
            matrixOperations.subtract(1.0, output)                // Compute (1 - output) for sigmoid derivative
        )
    );
    
    // Compute hidden layer error delta1
    double[] delta1 = matrixOperations.elementWiseMultiply(      // Element-wise multiplication for delta1
        matrixOperations.multiply(                                // Matrix multiplication to propagate delta2 back to hidden layer
            matrixOperations.transpose(W2),                       // Transpose W2 to align dimensions
            delta2                                               // Delta from the output layer
        ),
        matrixOperations.elementWiseMultiply(                      // Multiply by sigmoid derivative
            A1,                                                   // Activations from the hidden layer
            matrixOperations.subtract(1.0, A1)                    // Compute (1 - A1) for sigmoid derivative
        )
    );
    
    // Compute gradients for W2 and B2
    double[][] W2Gradient = new double[outputSize][hiddenSize];    // Initialize gradient matrix for W2
    double[] B2Gradient = new double[outputSize];                  // Initialize gradient vector for B2
    
    // Loop over each neuron in the output layer to compute gradients
    for (int i = 0; i < outputSize; i++) {                         // Iterate through output neurons
        for (int j = 0; j < hiddenSize; j++) {                     // Iterate through hidden neurons
            W2Gradient[i][j] = delta2[i] * A1[j];                  // Compute gradient for W2[i][j] = delta2[i] * activation of hidden neuron j
        }
        B2Gradient[i] = delta2[i];                                 // Gradient for B2[i] is delta2[i]
    }
    
    // Compute gradients for W1 and B1
    double[][] W1Gradient = new double[hiddenSize][inputSize];     // Initialize gradient matrix for W1
    double[] B1Gradient = new double[hiddenSize];                  // Initialize gradient vector for B1
    
    // Loop over each neuron in the hidden layer to compute gradients
    for (int i = 0; i < hiddenSize; i++) {                        // Iterate through hidden neurons
        for (int j = 0; j < inputSize; j++) {                     // Iterate through input neurons
            W1Gradient[i][j] = delta1[i] * input[j];               // Compute gradient for W1[i][j] = delta1[i] * input[j]
        }
        B1Gradient[i] = delta1[i];                                 // Gradient for B1[i] is delta1[i]
    }
    
    // Return the computed gradients encapsulated in a GradientValues object
    return new GradientValues(W1Gradient, W2Gradient, B1Gradient, B2Gradient);  // Return gradients for W1, W2, B1, B2
}

    
    /*6.) The "accumulateGradients(GradientValues gradients, ...)" method accumulates gradients over a minibatch, generalizing the gradient accumulation process for larger datasets.*/

    //Accumulates gradients from each sample in the minibatch.
    private void accumulateGradients(GradientValues gradients,
    double[][] W1GradientSum, double[][] W2GradientSum,
    double[] B1GradientSum, double[] B2GradientSum) {
        // Accumulate gradients for W1
        for (int i = 0; i < hiddenSize; i++) {                      // Loop over hidden neurons
            for (int j = 0; j < inputSize; j++) {                   // Loop over input neurons
                W1GradientSum[i][j] += gradients.W1Gradient[i][j];  // Add gradient to accumulator
            }
        }
        
        // Accumulate gradients for W2
        for (int i = 0; i < outputSize; i++) {                      // Loop over output neurons
            for (int j = 0; j < hiddenSize; j++) {                  // Loop over hidden neurons
                W2GradientSum[i][j] += gradients.W2Gradient[i][j];  // Add gradient to accumulator
            }
        }
        
        // Accumulate gradients for B1
        for (int i = 0; i < hiddenSize; i++) {                      // Loop over hidden neurons
            B1GradientSum[i] += gradients.B1Gradient[i];            // Add gradient to accumulator
        }
        
        // Accumulate gradients for B2
        for (int i = 0; i < outputSize; i++) {                      // Loop over output neurons
            B2GradientSum[i] += gradients.B2Gradient[i];            // Add gradient to accumulator
        }
    }
    
    /*7.) The "updateWeightsAndBiasesFromGradients(...)" method updates weights and biases using accumulated gradients after processing a minibatch; mirrors update mechanism from smallNeuralNetwork but scaled for larger matrices.*/
    //Updates weights and biases using learning rate and accumulated gradients after processing a minibatch.
    private void updateWeightsAndBiasesFromGradients(double[][] W1GradientSum, double[][] W2GradientSum,
    double[] B1GradientSum, double[] B2GradientSum,
    int minibatchSize) {
        double learningRate = eta / minibatchSize;  // Adjust learning rate based on minibatch size
        
        // Update W1 and B1
        for (int i = 0; i < hiddenSize; i++) {      // Loop over hidden neurons
            for (int j = 0; j < inputSize; j++) {   // Loop over input neurons
                W1[i][j] -= learningRate * W1GradientSum[i][j];   // Update weight W1[i][j]
            }
            B1[i] -= learningRate * B1GradientSum[i];             // Update bias B1[i]
        }
        
        // Update W2 and B2
        for (int i = 0; i < outputSize; i++) {      // Loop over output neurons
            for (int j = 0; j < hiddenSize; j++) {  // Loop over hidden neurons
                W2[i][j] -= learningRate * W2GradientSum[i][j];   // Update weight W2[i][j]
            }
            B2[i] -= learningRate * B2GradientSum[i];             // Update bias B2[i]
        }
    }
    
    
    /*8.) The "displayEpochStatistics(int epochNumber, List<double[]> data, List<double[]> labels)" method displays statistics after each epoch or when requested.*/
    //Displays statistics after each epoch or when requested.
    public void displayEpochStatistics(int epochNumber, List<double[]> data, List<double[]> labels) {
        // Initialize counters for each digit
        int[] correctCounts = new int[10];   // Array to store correct prediction counts for each digit
        int[] totalCounts = new int[10];     // Array to store total counts for each digit
        
        for (int i = 0; i < data.size(); i++) {     // Loop over all data samples
            double[] prediction = forward(data.get(i));     // Get the network's prediction
            int predictedLabel = getPrediction(prediction); // Determine predicted label
            int actualLabel = getPrediction(labels.get(i)); // Get actual label from one-hot encoded vector
            
            totalCounts[actualLabel]++;                  // Increment total count for the actual label
            if (predictedLabel == actualLabel) {         // Check if prediction is correct
                correctCounts[actualLabel]++;            // Increment correct count for the label
            }
        }
        
        // Display per-digit statistics
        for (int i = 0; i < 10; i++) {   // Loop over digits 0-9
            System.out.println("Digit " + i + ": " + correctCounts[i] + "/" + totalCounts[i]);  // Print correct/total for each digit
        }
        
        // Display overall accuracy
        int totalCorrect = Arrays.stream(correctCounts).sum();   // Sum of correct predictions
        int totalSamples = Arrays.stream(totalCounts).sum();     // Total number of samples
        double accuracy = (double) totalCorrect / totalSamples * 100;   // Calculate accuracy percentage
        System.out.println("Accuracy: " + totalCorrect + "/" + totalSamples + " = " + String.format("%.3f", accuracy) + "%");  // Print overall accuracy
    }
    
    //Runs the network on testing data, displaying images and labels.
    public void runNetworkOnTestingData(List<double[]> data, List<double[]> labels) {
        for (int i = 0; i < data.size(); i++) {    // Loop over all testing data samples
            double[] output = forward(data.get(i));    // Get the network's prediction
            int predictedLabel = getPrediction(output);   // Determine predicted label
            int actualLabel = getPrediction(labels.get(i));   // Get actual label
            
            // Display image as ASCII art
            System.out.println("\nTesting Case #" + (i + 1) + ":");   // Print case number
            displayImageAsASCII(data.get(i));     // Display the image in ASCII art
            
            // Display the classification results
            System.out.println("\nCorrect classification = " + actualLabel +
            "  Network Output = " + predictedLabel +
            (predictedLabel == actualLabel ? "  Correct." : "  Incorrect."));  // Indicate if the prediction is correct
            
            // Allow user to continue or return to main menu
            System.out.print("\nEnter 1 to continue. All other values return to main menu: ");
            try {
                int choice = scanner.nextInt();    // Read user's choice
                if (choice != 1) {
                    break;    // Exit to main menu if choice is not 1
                }
            } catch (InputMismatchException e) {   // Handle invalid input
                System.out.println("Invalid input. Returning to main menu.");
                scanner.nextLine();   // Clear the invalid input
                break;                // Exit to main menu
            }
        }
    }
    
    //Converts the network's output activations to a predicted label.
    private int getPrediction(double[] output) {
        int prediction = 0;        // Initialize prediction index
        double max = output[0];    // Initialize max activation value
        
        // Find the index with the highest activation
        for (int i = 1; i < output.length; i++) {   // Loop over output activations
            if (output[i] > max) {
                max = output[i];    // Update max value
                prediction = i;     // Update prediction index
            }
        }
        return prediction;    // Return the predicted label
    }
    
    //Displays misclassified images from the testing data.
    public void displayMisclassifiedTestingImages(List<double[]> data, List<double[]> labels) {
        int misclassifiedCount = 0;    // Counter for misclassified images
        for (int i = 0; i < data.size(); i++) {   // Loop over testing data
            double[] output = forward(data.get(i));     // Get the network's prediction
            int predictedLabel = getPrediction(output); // Determine predicted label
            int actualLabel = getPrediction(labels.get(i));   // Get actual label
            
            if (predictedLabel != actualLabel) {    // Check if prediction is incorrect
                misclassifiedCount++;               // Increment misclassified count
                System.out.println("\nMisclassified - Predicted: " + predictedLabel + " | Actual: " + actualLabel);   // Display labels
                displayImageAsASCII(data.get(i));   // Display the misclassified image
                
                // Allow user to continue or return to main menu
                System.out.print("\nEnter 1 to continue. All other values return to main menu: ");
                try {
                    int choice = scanner.nextInt();    // Read user's choice
                    if (choice != 1) {
                        break;    // Exit to main menu if choice is not 1
                    }
                } catch (InputMismatchException e) {   // Handle invalid input
                    System.out.println("Invalid input. Returning to main menu.");
                    scanner.nextLine();   // Clear the invalid input
                    break;                // Exit to main menu
                }
            }
        }
        if (misclassifiedCount == 0) {    // Check if no misclassifications were found
            System.out.println("No misclassified images found.");
        }
    }
    
    
    //Saves the network's weights and biases to a file.
    //Writes the current weights (W1, W2) and biases (B1, B2) of the neural network to a file (e.g., network_state.txt) in a structured format. It iterates through each weight matrix and bias vector, writing their values line by line, ensuring that all parameters are accurately saved for future retrieval and restoration of the network's state.
    public void saveNetworkState() {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter("network_state.txt"))) {  // Open file for writing
            // Save weights W1
            for (double[] row : W1) {    // Loop over rows of W1
                for (double weight : row) {   // Loop over weights in the row
                    writer.write(weight + " ");   // Write weight followed by a space
                }
                writer.newLine();   // New line after each row
            }
            writer.newLine();   // Empty line to separate sections
            
            // Save biases B1
            for (double bias : B1) {   // Loop over biases in B1
                writer.write(bias + " ");   // Write bias followed by a space
            }
            writer.newLine();
            writer.newLine();
            
            // Save weights W2
            for (double[] row : W2) {    // Loop over rows of W2
                for (double weight : row) {   // Loop over weights in the row
                    writer.write(weight + " ");   // Write weight followed by a space
                }
                writer.newLine();   // New line after each row
            }
            writer.newLine();
            
            // Save biases B2
            for (double bias : B2) {   // Loop over biases in B2
                writer.write(bias + " ");   // Write bias followed by a space
            }
            writer.newLine();
            
            System.out.println("Network state saved to 'network_state.txt'.");   // Confirmation message
        } catch (IOException e) {   // Handle exceptions
            System.err.println("Error saving network state: " + e.getMessage());   // Error message
        }
    }
    
    
    //Loads the network's weights and biases from a file.
    public void loadNetwork() {
        try (BufferedReader reader = new BufferedReader(new FileReader("network_state.txt"))) {   // Open file for reading
            String line;             // Variable to hold each line read from the file
            String[] weightStrings;  // Array to hold string representations of weights
            
            // Load weights W1
            for (int i = 0; i < W1.length; i++) {   // Loop over rows of W1
                line = reader.readLine();           // Read a line from the file
                if (line != null && !line.trim().isEmpty()) {    // Check if line is not null or empty
                    weightStrings = line.trim().split("\\s+");   // Split the line by whitespace
                    for (int j = 0; j < W1[0].length; j++) {     // Loop over columns
                        W1[i][j] = Double.parseDouble(weightStrings[j]);   // Parse and assign weight
                    }
                }
            }
            reader.readLine();   // Skip empty line
            
            // Load biases B1
            line = reader.readLine();   // Read biases for B1
            if (line != null && !line.trim().isEmpty()) {
                weightStrings = line.trim().split("\\s+");   // Split the line by whitespace
                for (int i = 0; i < B1.length; i++) {        // Loop over biases
                    B1[i] = Double.parseDouble(weightStrings[i]);   // Parse and assign bias
                }
            }
            reader.readLine();   // Skip empty line
            
            // Load weights W2
            for (int i = 0; i < W2.length; i++) {   // Loop over rows of W2
                line = reader.readLine();           // Read a line from the file
                if (line != null && !line.trim().isEmpty()) {
                    weightStrings = line.trim().split("\\s+");   // Split the line by whitespace
                    for (int j = 0; j < W2[0].length; j++) {     // Loop over columns
                        W2[i][j] = Double.parseDouble(weightStrings[j]);   // Parse and assign weight
                    }
                }
            }
            reader.readLine();   // Skip empty line
            
            // Load biases B2
            line = reader.readLine();   // Read biases for B2
            if (line != null && !line.trim().isEmpty()) {
                weightStrings = line.trim().split("\\s+");   // Split the line by whitespace
                for (int i = 0; i < B2.length; i++) {        // Loop over biases
                    B2[i] = Double.parseDouble(weightStrings[i]);   // Parse and assign bias
                }
            }
            
            System.out.println("Network state loaded from 'network_state.txt'.");   // Confirmation message
        } catch (IOException e) {   // Handle exceptions
            System.err.println("Error loading network state: " + e.getMessage());   // Error message
        }
    }
    
    /*10.) The "displayImageAsASCII(double[] pixels)" method displays an image represented by pixel values as ASCII art*/
    //Displays an image represented by pixel values as ASCII art.
    private void displayImageAsASCII(double[] pixels) {
        int width = 28; // MNIST images are 28x28
        for (int i = 0; i < pixels.length; i++) {//loop over all pixels
            if (i % width == 0) System.out.println(); // New line after each row
            System.out.print(pixelToASCII(pixels[i]));//print ASCII character for the pixel
        }
        System.out.println();
    }
    
    /*
    11.) The "pixelToASCII(double pixelValue)" method converts a pixel value to an ASCII character based on intensity; maps higher pixel values to `'@'`, making digits stand out.
    * Converts a pixel value to an ASCII character based on intensity, mapping 
    * higher pixel values (brighter pixels) to '@' symbols,
    * making the digits stand out in the ASCII representation.
    * The background pixels (darker pixels) are represented with spaces.
    */
    private String pixelToASCII(double pixelValue) {
        if (pixelValue > 0.8) return "@";//brightest pixels
        else if (pixelValue > 0.6) return "o";
        else if (pixelValue > 0.4) return ":";
        else if (pixelValue > 0.2) return ".";
        else return " ";//darkest pixels (background)
    }
    
    /*13.) "GradientValues" holds gradient values for weights and biases; improves code organization compared to smallNeuralNetwork, which handles gradients directly.*/
    //Inner class to hold gradient values for weights and biases.
    public class GradientValues {
        public double[][] W1Gradient;//gradients for weights between input and hidden layer
        public double[][] W2Gradient;// gradients for weights between hidden and output layer
        public double[] B1Gradient;//gradients for biases of hidden layer
        public double[] B2Gradient; //gradients for biases of output layer
        
        //Constructor for GradientValues.
        public GradientValues(double[][] W1Gradient, double[][] W2Gradient, double[] B1Gradient, double[] B2Gradient) {
            this.W1Gradient = W1Gradient;
            this.W2Gradient = W2Gradient;
            this.B1Gradient = B1Gradient;
            this.B2Gradient = B2Gradient;
        }
    }
}

