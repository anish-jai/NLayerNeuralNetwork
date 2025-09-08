/*
 * This class represents an N-Layer neural network with backpropogation, meaning it can have any number of layers and nodes within
 * each layer. The network can either run based on predetermined weights or train using backpropogation, optimizing 
 * the values of the weights under a certain error threshold. It can also load weights or use predetermined weight values. The 
 * configuration is done through an external file.
 * 
 * 
 * @author Anish Jain
 * @version 4.30.24
 * 
 * Table of Contents:
 * public static void main(String[] args)
 * public static void setConfigParams()
 * public static int maxArrayDim()
 * public static double sigmoid(double x)
 * public static double derivativeSigmoid(double x)
 * public static double activationFunction(double num)
 * public static double derivativeActivationFunction(double num)
 * public static void echoConfigParams()
 * public static void allocateArrayMemory()
 * public static void populateArrays()
 * public static void populateWeightsRandom()
 * public static void train()
 * public static void runSingleOutput()
 * public static void runNetworkOutput()
 * public static void calculateError()
 * public static void reportResults()
 * public static double generateRandWeight()
 * public static void saveWeights()
 * public static void loadWeights()
 * 
 */
import java.io.*;
import java.util.Properties;                 // used to read in config file

public class NLayer
{
/*
 * Declares all const variables for network config.
 */
   public static int maxIters;
   public static double lambda;              // learning factor
   public static double errorThreshold;
   public static double randomMinThreshold;  // minimum value when randomly generating weights
   public static double randomMaxThreshold;  // maximum value when randomly generating weights
   public static boolean willTrain;          // network will optimize weights using gradient descent if true
   public static boolean useRandomWeights;
   public static boolean willSaveWeights;
   public static boolean useLoadedWeights;
   public static int numTestCases;
   public static double defaultWeightVal;


/*
 * Declare all variables for internal use in other functions and loops.
 */
   public static boolean isTrainingDone;
   public static double averageError;
   public static double totalError;
   public static String reasonToStopTraining;      // string that contains the reason why the program terminated training
   public static int m, i;                         // iterator variables for input and output layers
   public static int alpha, beta, gamma;
   public static int tCaseIter;                    // iterator variable for going through test cases
   public static int totalIters;                   // total number of iterations 
   public static String[] netConfigStr;
   public static String configFileName;
   public static String weightsFileName;
   public static String tCaseActivationFileName;
   public static int keepAlive;
   public static int runningNum;
   
/*
* Declare all arrays, including arrays for test cases, omegas, psi values, and weight calculations.
*/
   public static double[][] testCases;             // array with the truth table for test cases
   public static double[][] targetOutput;          // array with the expected outputs for test cases
   public static double[][] calculatedOutputs;     // array with the generated outputs for test cases
   public static double[][] activations;           // one activations array for all layers
   public static double[][][] weights;             // one weight array for all layers
   public static double[][] theta;                 // one theta array for all layers
   public static double[][] psi;
   public static int[] nLayers;

   public static long startingTime;
   public static long endingTime;
   public static long elapsedTimeMilli;
   public static Properties testActivations;
   public static Properties properties;

   public static int n;
   public static int numLayers;
   public static int maxLayerSize;
   public static int inLayer, outLayer;


/*
 * Useful ANSI escape codes for formatting
 */
   public static String ANSI_BOLD = "\u001B[1m";
   public static String ANSI_RESET = "\u001B[0m";

/*
 * The main method that will either run or train the neural network. The method sets and prints the config parameters 
 * needed for the network configuration.
 * 
 * @param args array of command line arguments
 */
   public static void main(String[] args)
   {
      startingTime = System.nanoTime();
      configFileName = "ImageProcessingConfig.txt";
      if (args.length > 0)
      {
         configFileName = args[0];
      }

      setConfigParams();
      echoConfigParams();
      allocateArrayMemory();
      populateArrays();

      if (willTrain)
      {
         train();
         runNetworkOutput();
      }
      else 
      {
         runNetworkOutput();
      }
      reportResults();

      if (willSaveWeights)
      {
         saveWeights();
      }

/*
 * Calculates the amount of time needed to run this network.
 */
      endingTime = System.nanoTime();
      elapsedTimeMilli = (endingTime - startingTime) / 1000000;         // Convert nanoseconds to milliseconds
      System.out.println("Total elapsed time: " + ANSI_BOLD + elapsedTimeMilli + "ms." + ANSI_RESET);
      System.out.println();

   } // main(String[] args)

/*
 * This method sets basic parameters for this network using a config file, including but not limited to
 * the number of inputs, number of input activations, thresholds for the random weight generation,
 * the max number of iterations, the learning factor, and whether the network is training or not.
 */
   public static void setConfigParams()
   {
      try 
      {
         properties = new Properties();
         properties.load(new FileInputStream(configFileName));

         numTestCases = Integer.parseInt(properties.getProperty("numTestCases"));
         runningNum = Integer.parseInt(properties.getProperty("runningNum"));
         defaultWeightVal = Double.parseDouble(properties.getProperty("defaultWeightVal"));
         netConfigStr = (properties.getProperty("netConfig")).split("-");
         numLayers = netConfigStr.length;
         nLayers = new int[numLayers];
         for (alpha = inLayer; alpha < numLayers; alpha++) 
         {
            nLayers[alpha] = Integer.parseInt(netConfigStr[alpha]);
         }         

         errorThreshold = Double.parseDouble(properties.getProperty("errorThreshold"));
         averageError = Double.parseDouble(properties.getProperty("averageError"));
         maxIters = Integer.parseInt(properties.getProperty("maxIters"));
         lambda = Double.parseDouble(properties.getProperty("lambda"));
         keepAlive = Integer.parseInt(properties.getProperty("keepAlive"));

         randomMinThreshold = Double.parseDouble(properties.getProperty("randomMinThreshold"));
         randomMaxThreshold = Double.parseDouble(properties.getProperty("randomMaxThreshold"));

         willTrain = Boolean.parseBoolean(properties.getProperty("willTrain"));
         useRandomWeights = Boolean.parseBoolean(properties.getProperty("useRandomWeights"));
         willSaveWeights = Boolean.parseBoolean(properties.getProperty("willSaveWeights"));
         useLoadedWeights = Boolean.parseBoolean(properties.getProperty("useLoadedWeights"));
         weightsFileName = properties.getProperty("weightsFileName");
         tCaseActivationFileName = properties.getProperty("tCaseActivationFileName");
      } // try
      catch (IOException e) 
      {
         e.printStackTrace();
      }
      inLayer = 0;
      outLayer = numLayers - 1;
      reasonToStopTraining = "";
   } // setConfigParams()

/**
 * Gives the largest dimension needed for necessary arrays, such as theta or weight arrays.
 * @return int max array dimension 
 */
   public static int maxArrayDim()
   {
      int retVal = nLayers[inLayer];
      for (alpha = inLayer + 1; alpha < numLayers; alpha++)
      {
         if (nLayers[alpha] > retVal)
         {
            retVal = nLayers[alpha];
         }
      }
      return retVal;
   } // maxArrayDim()



/**
 * Gives the largest dimension needed for necessary arrays, such as theta or weight arrays.
 * @return int max array dimension 
 */
public static int secondMaxArrayDim()
{
   int retVal = nLayers[inLayer+1];
   for (alpha = inLayer + 1; alpha < numLayers; alpha++)
   {
      if (nLayers[alpha] > retVal)
      {
         retVal = nLayers[alpha];
      }
   }
   return retVal;
} // maxArrayDim()
/*
 * The sigmoid function is an activation function. It is bounded between 0 and 1.
 * 
 * @param x the value to plug into the function
 * @return double the calculated value for sigmoid(x)
 */
   public static double sigmoid(double x)
   {
      return (1.0 / (1.0 + Math.exp(-x)));
   }

/*
 * Derivative of the sigmoid function. Useful for intermediate calculations and weight optimizations.
 * The first derivative of sigmoid has a shape that is bell shaped.
 * 
 * @param x the value to plug into the function
 * @return double the calculated value for sigmoid'(x)
 */
   public static double derivativeSigmoid(double x)
   {
      double val = sigmoid(x);
      return val * (1.0 - val);
   }

/*
 * Activation function used throughout this neural network.
 * 
 * @param num value to plug into the activation function
 * @return double the calculated value of f(x), where f is the activation function
 * 
 */
   public static double activationFunction(double num)
   {
      return sigmoid(num);
   }

/*
 * Derivative of the activation function for this neural network.
 * 
 * @param num value to plug into the function
 * @return double the calculated value of f'(x), where f' is the derivative of the activation function
 * 
 */
   public static double derivativeActivationFunction(double num)
   {
      return derivativeSigmoid(num);
   }

/*
 * Prints out the parameters set in the config. 
 */
   public static void echoConfigParams()
   {
      System.out.println("----------------------------------------------");      
      System.out.print("N-Layer Network Configuration= ");

      for (alpha = inLayer; alpha < numLayers - 1; alpha++)
      {
         System.out.print(nLayers[alpha] + "-");
      }

      System.out.println(nLayers[outLayer]);
      System.out.println("Config File: " + configFileName);
      
      if (useRandomWeights)
      {
         System.out.println("Weights will be randomly generated. Bounds for Minimum and Maximum Weight Generation = ("
               + randomMinThreshold + " , " + randomMaxThreshold + ")");
      }
      else if (useLoadedWeights)
      {
         System.out.println("Weights will be loaded from " + weightsFileName + ".");
      }
      else
      {
         System.out.println("Manual weights will be used.");
      }

      if (willSaveWeights)
      {
         System.out.println("Array data will be written to " + weightsFileName);
      }

      if (willTrain)
      {
         System.out.println("Maximum Number of Iterations = " + maxIters);
         System.out.println("Error Threshold to Stop Training = " + errorThreshold);
         System.out.println("Learning Factor = " + lambda);
         System.out.println("Training will now begin.");
      }
      else
      {
         System.out.println("Model is only using set #" + runningNum + " of images while running.");
         System.out.println("Model is now Running without training.");
      }
      System.out.println("----------------------------------------------");      
   } // echoConfigParams()

/*
 * Allocates space for test case, activations, output, target output, and weight arrays.
 * If the model is currently training, also allocates memory for the weight optimization arrays.
 */
   public static void allocateArrayMemory()
   {
      testCases = new double[numTestCases][nLayers[inLayer]];
      targetOutput = new double[numTestCases][nLayers[outLayer]];
      calculatedOutputs = new double[numTestCases][nLayers[outLayer]];
      
      maxLayerSize = maxArrayDim();
      activations = new double[numLayers][maxLayerSize];
      weights = new double[numLayers][maxLayerSize][secondMaxArrayDim()];
      
      if (willTrain)
      {
         theta = new double[numLayers][maxLayerSize];
         psi = new double[numLayers][maxLayerSize]; 
      }
   } // allocateArrayMemory()

/*
 * Inserts training values into test case and target output arrays by reading in the config file. 
 * Then, either loads, randomizes, or manually sets the weights based on the associated boolean values.
 */
   public static void populateArrays()
   {
      try 
      {
         BufferedReader testActivations;
         String line;
         String[] values;
         if (willTrain)
         {
            for (tCaseIter = 0; tCaseIter < numTestCases; tCaseIter++) 
            {
               tCaseActivationFileName = "TestCases/" + (tCaseIter % 5 + 1) + "-" + (tCaseIter / (numTestCases / 5) + 1) + ".txt";
               // System.out.println("Getting activations from " + tCaseActivationFileName);
               testActivations = new BufferedReader(new FileReader(tCaseActivationFileName));
               line = testActivations.readLine();
               values = line.trim().split("\\s+");
               for (m = 0; m < nLayers[inLayer]; m++) {
                  testCases[tCaseIter][m] = Double.parseDouble(values[m]);
                  // if (tCaseIter == 1 && testCases[tCaseIter][m] != 0.0)
                  //    System.out.print(testCases[tCaseIter][m]);
               }
               testActivations.close();

            }
         }
         else
         {
            for (tCaseIter = 0; tCaseIter < numTestCases/5; tCaseIter++) 
            {
               tCaseActivationFileName = "TestCases/" + runningNum + "-" + (tCaseIter + 1) + ".txt";
               // System.out.println(tCaseActivationFileName);
               testActivations = new BufferedReader(new FileReader(tCaseActivationFileName));
               line = testActivations.readLine();
               values = line.trim().split("\\s+");
               for (m = 0; m < nLayers[inLayer]; m++) {
                  testCases[tCaseIter][m] = Double.parseDouble(values[m]);
                  // if (tCaseIter == 1 && testCases[tCaseIter][m] != 0.0)
                  //    System.out.print(testCases[tCaseIter][m]);
               }
               testActivations.close();

            }
         }
      } // try
      catch (IOException e) 
      {
         e.printStackTrace();
      }

      try {
         properties.load(new FileInputStream(configFileName));
      }
      catch (IOException e) 
      {
         // System.out.println("didnt work");
         e.printStackTrace();
      }
      for (tCaseIter = 0; tCaseIter < numTestCases; tCaseIter++) 
      {
         if (willTrain)
         {
            for (i = 0; i < nLayers[outLayer]; i++) {
               targetOutput[tCaseIter][i] = Double
                     .parseDouble(properties.getProperty("targetOutput_" + tCaseIter + "_" + i));
            }
         }
         else 
         {
            for (i = 0; i < nLayers[outLayer]; i++) {
               targetOutput[tCaseIter][i] = Double
                     .parseDouble(properties.getProperty("targetOutput_" + (tCaseIter * 5 + 1) + "_" + i));
            }
         }
      }

      if (useRandomWeights)
      {
         populateWeightsRandom();
      }
      else if (useLoadedWeights)
      {
         loadWeights();
      }
      else      // sets weights with default value
      {
         for (alpha = 1; alpha < numLayers; alpha++)
         {
            for (gamma = 0; gamma < nLayers[alpha - 1]; gamma++)
            {
               for (beta = 0; beta < nLayers[alpha]; beta++)
               {
                  weights[alpha][gamma][beta] = defaultWeightVal;
               }
            }
         } // for (alpha = 1; alpha < numLayers; alpha++)

      } // else if (useLoadedWeights)...else

   } // populateArrays()
   
/*
 * Randomly generates weight values within the min and max threshold values.
 */
   public static void populateWeightsRandom()
   {
      for (alpha = 1; alpha < numLayers; alpha++)
      {
         for (gamma = 0; gamma < nLayers[alpha - 1]; gamma++)
         {
            for (beta = 0; beta < nLayers[alpha]; beta++)
            {
               weights[alpha][gamma][beta] = generateRandWeight();
            }
         }
      } // for (alpha = 1; alpha < numLayers; alpha++)
   } // populateWeightsRandom()

/*
 * Trains the model by optimizing weights, applying a delta each iteration. 
 * This method uses trains with backpropogation until either the max iterations or error threshold 
 * has been reached. 
 */
   public static void train()
   {
      totalIters = 0;
      isTrainingDone = false;

      while (!isTrainingDone)
      {
         totalError = 0.0;
         for (tCaseIter = 0; tCaseIter < numTestCases; tCaseIter++) // loop through all test cases
         {
            for (m = 0; m < nLayers[inLayer]; m++)                  // place test cases in input activations array
            {
               activations[inLayer][m] = testCases[tCaseIter][m];
            }

            for (alpha = 1; alpha < numLayers - 1; alpha++)
            {
               for (beta = 0; beta < nLayers[alpha]; beta++)
               {
                  theta[alpha][beta] = 0.0;
                  for (gamma = 0; gamma < nLayers[alpha - 1]; gamma++) 
                  {
                     theta[alpha][beta] += activations[alpha - 1][gamma] * weights[alpha][gamma][beta];
                  }
                  activations[alpha][beta] = activationFunction(theta[alpha][beta]);
               }
            } // for (alpha = 1; alpha < numLayers; alpha++)

            alpha = outLayer;
            for (beta = 0; beta < nLayers[alpha]; beta++)           // separate loop for output layer
            {
               theta[alpha][beta] = 0.0;
               for (gamma = 0; gamma < nLayers[alpha - 1]; gamma++) 
               {
                  theta[alpha][beta] += activations[alpha - 1][gamma] * weights[alpha][gamma][beta];
               }
               activations[alpha][beta] = activationFunction(theta[alpha][beta]);
               psi[alpha][beta] = 
                  (targetOutput[tCaseIter][beta] - activations[alpha][beta]) * derivativeActivationFunction(theta[alpha][beta]);
            } // for (beta = 0; beta < nLayers[alpha]; beta++)
  
/*
 * Performs weight optimization using backpropogation.
 */
            double omegaSum;
            for (alpha = outLayer - 1; alpha > inLayer; alpha--)
            {
               for (gamma = 0; gamma < nLayers[alpha]; gamma++)
               {
                  omegaSum = 0;
                  for (beta = 0; beta < nLayers[alpha + 1]; beta++) 
                  {
                     omegaSum += psi[alpha + 1][beta] * weights[alpha + 1][gamma][beta];
                     weights[alpha + 1][gamma][beta] += lambda * activations[alpha][gamma] * psi[alpha + 1][beta];
                  }

                  psi[alpha][gamma] = omegaSum * derivativeActivationFunction(theta[alpha][gamma]);
               }
            } // for (alpha = outLayer - 1; alpha > 0; alpha--) 

            alpha = inLayer;
            for (gamma = 0; gamma < nLayers[alpha]; gamma++)
            {
               for (beta = 0; beta < nLayers[alpha + 1]; beta++) 
               {
                  weights[alpha + 1][gamma][beta] += lambda * activations[alpha][gamma] * psi[alpha + 1][beta];
               }
            }

            runSingleOutput();
            calculateError();
         } // for (tCaseIter = 0; tCaseIter < numTestCases; tCaseIter++)

         averageError = totalError / (double) numTestCases;
         totalIters++;
         isTrainingDone = totalIters >= maxIters || averageError <= errorThreshold;

         if (keepAlive != 0 && totalIters % keepAlive == 0)
         {
            System.out.printf("Iteration %d, Error = %f\n", totalIters, averageError);
         }

      } // while (!isTrainingDone)
   } // train()

/*
 * Runs a single case for its output. There is no need for truth tables to run this method.
 */
   public static void runSingleOutput()
   {
      double thetaSum = 0.0;
      for (alpha = 1; alpha < numLayers; alpha++)
      {
         for (beta = 0; beta < nLayers[alpha]; beta++)
         {
            thetaSum = 0.0;
            for (gamma = 0; gamma < nLayers[alpha - 1]; gamma++) 
            {
               thetaSum += activations[alpha - 1][gamma] * weights[alpha][gamma][beta];
            }
            activations[alpha][beta] = activationFunction(thetaSum);
         }
      } // for (alpha = 1; alpha < numLayers; alpha++)

   } // runSingleOutput()

/**
 * Runs network for output by filling activations and running each case individually.
 * Does not require target outputs, simply runs the network.
 */
   public static void runNetworkOutput()
   {
      for (tCaseIter = 0; tCaseIter < numTestCases; tCaseIter++)
      {
         for (m = 0; m < nLayers[inLayer]; m++) // place test cases in input activations array
         {
            activations[inLayer][m] = testCases[tCaseIter][m];
         }

         runSingleOutput();

         for (i = 0; i < nLayers[outLayer]; i++) 
         {
            calculatedOutputs[tCaseIter][i] = activations[outLayer][i];
         }
      } // for (tCaseIter = 0; tCaseIter < numTestCases; tCaseIter++)

   } // runNetworkOutput()

/*
 * Finds error across test cases by finding the sum of all individual errors for the test cases.
 */
   public static void calculateError()
   {
      for (i = 0; i < nLayers[outLayer]; i++)
      {
         totalError += Math.pow((targetOutput[tCaseIter][i] - activations[outLayer][i]), 2) * 0.5;
      }
   } // calculateError()

/*
 * Reports results of training or running, by printing the calculated and expected values for the test cases.
 * If the model just finished training, the method also reports the reason why.
 */
   public static void reportResults()
   {
      if (totalIters >= maxIters)
      {
         reasonToStopTraining = "The maximum number of iterations, " + maxIters + ", has been reached. ";
      }

      if (averageError <= errorThreshold)
      {
         reasonToStopTraining += "The program achieved the error cutoff in " + totalIters + " iterations. ";
      }

      reasonToStopTraining += "Thus, the program ended.";

      System.out.println("----------------------------------------------");
      if (willTrain)
      {
         System.out.println("The network has stopped training. ");
         System.out.println(reasonToStopTraining);
         System.out.println("The total number of iterations is " + totalIters + ".");
         System.out.println("Average error of the model is " + averageError + " while the error threshold was " + errorThreshold);
      }
      else
      {
         System.out.println("The network has finished running. ");
      }

      System.out.println("----------------------------------------------");

/*
 * Prints truth tables to compare expected and calculated values.
 */     
      System.out.println("Truth Table with expected and calculated  values.");
      System.out.println("Inputs are labeled I, Target/Expected Outputs are marked T, and Calculated Outputs are marked F.\n");
    
      // for (m = 0; m < nLayers[inLayer]; m++)
      // {
      //    System.out.print(ANSI_BOLD + " I#" + (m+1) + " |" + ANSI_RESET);     // ANSI_BOLD bolds the text for headers
      // }
      
      for (i = 0; i < nLayers[outLayer]; i++)
      {
         System.out.print(ANSI_BOLD + "  T#" + (i+1)+ "  |" + ANSI_RESET);
      }
      
      for (i = 0; i < nLayers[outLayer]; i++)
      {
         System.out.print(ANSI_BOLD + "  F#" + (i+1)+  "  |" + ANSI_RESET);
      }
      
      for (tCaseIter = 0; tCaseIter < numTestCases; tCaseIter++)
      {
         System.out.println();
         // for (m = 0; m < nLayers[inLayer]; m++) 
         // {
         //    System.out.print(" " + testCases[tCaseIter][m] + " |");
         // }

         for (i = 0; i < nLayers[outLayer]; i++) 
         {
            System.out.printf(" %.3f", targetOutput[tCaseIter][i]);
            System.out.print(" |");
         }

         for (i = 0; i < nLayers[outLayer]; i++)
         {
            System.out.printf(" %.3f", calculatedOutputs[tCaseIter][i]);
            System.out.print(" |");
         }
      } // for (tCaseIter = 0; tCaseIter < numTestCases; tCaseIter++)
      System.out.println("\n----------------------------------------------");
      
   }  // reportResults()

/*
 * Generates a random weight as a baseline for training.
 * 
 * @return double randomly generated weight value, within the bounds of the min and max thresholds
 */
   public static double generateRandWeight()
   {
      return (double) (Math.random() * (randomMaxThreshold - randomMinThreshold) + randomMinThreshold);
   }  // generateRandWeight()

/*
 * Saves weights into a .txt file.
 */
   public static void saveWeights()
   {
      try (PrintWriter writer = new PrintWriter(new FileWriter(weightsFileName))) 
      {
         for (alpha = 1; alpha < numLayers; alpha++)
         {
            for (gamma = 0; gamma < nLayers[alpha - 1]; gamma++)
            {
               for (beta = 0; beta < nLayers[alpha]; beta++)
               {
                  writer.print(weights[alpha][gamma][beta] + " ");
               }
               writer.println();
            }
            writer.println();
         } // for (alpha = 1; alpha < numLayers; alpha++)

         System.out.println("Array data has been written to " + weightsFileName);

      } // try (PrintWriter writer = new PrintWriter(new FileWriter(weightsFileName))) 
      catch (IOException e) 
      {
         System.err.println("Error writing to file: " + e.getMessage());
      }
   } // saveWeights()
   
/*
 * Reads in and loads weights from a .txt file.
 */
   public static void loadWeights() 
   {
      try 
      {
         BufferedReader reader = new BufferedReader(new FileReader(weightsFileName));
         String[] line;
         for (alpha = 1; alpha < numLayers; alpha++)
         {
            for (gamma = 0; gamma < nLayers[alpha - 1]; gamma++)
            {
               line = reader.readLine().trim().split("\\s+");
               for (beta = 0; beta < nLayers[alpha]; beta++) 
               {
                  weights[alpha][gamma][beta] = Double.parseDouble(line[beta]);
               }
            }
            reader.readLine();
         } // for (alpha = 1; alpha < numLayers; alpha++)

         System.out.println("Array data has been loaded from " + weightsFileName);
         reader.close();

      } // try
      catch (IOException e) 
      {
         System.err.println("Error reading from file: " + e.getMessage());
      }
   } // loadWeights()

} // public class NLayer