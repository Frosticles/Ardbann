/*
  Ardbann.cpp - ARDuino Backpropogating Artificial Neural Network.
  Created by Peter Frost, February 9, 2017.
  Released into the public domain.
*/

#include "Ardbann.h"

Ardbann::Ardbann(uint16_t rawInputArray[], const uint16_t maxInput,
                 String outputArray[], const uint16_t numInputs,
                 const uint16_t numInputNeurons,
                 const uint16_t numHiddenNeurons, const uint8_t numHiddenLayers,
                 const uint16_t numOutputNeurons)
{
  float **outputLayerWeightTable = new float *[numOutputNeurons];
  float *outputLayerBiases = new float[numOutputNeurons];

  randomSeed(analogRead(3));
  // This pin should ideally be floating, change if using this pin

  for (uint8_t i = 0; i < numOutputNeurons; i++)
  {
    outputLayerWeightTable[i] = new float[numHiddenNeurons];
    outputLayerBiases[i] = ((float)random(-1000, 1000)) / 1000;
    for (uint16_t j = 0; j < numHiddenNeurons; j++)
    {
      outputLayerWeightTable[i][j] = ((float)random(-1000, 1000)) / 1000;
    }
  }

  float **hiddenLayerNeuronTable = new float *[numHiddenLayers];
  float **hiddenLayerNeuronBiasTable = new float *[numHiddenLayers];
  float ***hiddenLayerNeuronWeightLayerTable = new float **[numHiddenLayers];

  for (uint8_t i = 0; i < numHiddenLayers; i++)
  {
    hiddenLayerNeuronWeightLayerTable[i] = new float *[numHiddenNeurons];
    hiddenLayerNeuronTable[i] = new float[numHiddenNeurons];
    hiddenLayerNeuronBiasTable[i] = new float[numHiddenNeurons];

    for (uint16_t j = 0; j < numHiddenNeurons; j++)
    {
      hiddenLayerNeuronWeightLayerTable[i][j] = new float[numInputNeurons];
      hiddenLayerNeuronTable[i][j] = 0;
      hiddenLayerNeuronBiasTable[i][j] = ((float)random(-1000, 1000)) / 1000;

      for (uint16_t k = 0; k < numInputNeurons; k++)
      {
        hiddenLayerNeuronWeightLayerTable[i][j][k] =
            ((float)random(-1000, 1000)) / 1000;
      }
    }
  }

  network.numLayers = numHiddenLayers + 2;
  network.networkResponse = 0;

  network.inputLayer.numNeurons = numInputNeurons;
  network.inputLayer.numRawInputs = numInputs;
  network.inputLayer.neurons = new float[numInputNeurons];
  network.inputLayer.rawInputs = rawInputArray;
  network.inputLayer.maxInput = maxInput;
  network.inputLayer.groupThresholds = new uint16_t[numInputNeurons];
  network.inputLayer.groupTotal = new uint16_t[numInputNeurons];

  CalculateInputNeurons();

  network.outputLayer.numNeurons = numOutputNeurons;
  network.outputLayer.neurons = new float[numOutputNeurons];
  network.outputLayer.weightTable = outputLayerWeightTable;
  network.outputLayer.stringArray = outputArray;
  network.outputLayer.neuronBiasTable = outputLayerBiases;

  network.hiddenLayer.numNeurons = numHiddenNeurons;
  network.hiddenLayer.numLayers = numHiddenLayers;
  network.hiddenLayer.neuronTable = hiddenLayerNeuronTable;
  network.hiddenLayer.weightLayerTable = hiddenLayerNeuronWeightLayerTable;
  network.hiddenLayer.neuronBiasTable = hiddenLayerNeuronBiasTable;
}

Ardbann::Ardbann(const uint16_t maxInput, String outputArray[],
                 const uint16_t numInputNeurons,
                 const uint16_t numHiddenNeurons, const uint8_t numHiddenLayers,
                 const uint16_t numOutputNeurons)
{

  float **outputLayerWeightTable = new float *[numOutputNeurons];
  float *outputLayerBiases = new float[numOutputNeurons];

  for (uint8_t i = 0; i < numOutputNeurons; i++)
  {
    outputLayerWeightTable[i] = new float[numHiddenNeurons];
    outputLayerBiases[i] = ((float)random(-1000, 1000)) / 1000;
    for (uint16_t j = 0; j < numHiddenNeurons; j++)
    {
      outputLayerWeightTable[i][j] = ((float)random(-1000, 1000)) / 1000;
    }
  }

  float **hiddenLayerNeuronTable = new float *[numHiddenLayers];
  float **hiddenLayerNeuronBiasTable = new float *[numHiddenLayers];
  float ***hiddenLayerNeuronWeightLayerTable = new float **[numHiddenLayers];

  for (uint8_t i = 0; i < numHiddenLayers; i++)
  {
    hiddenLayerNeuronWeightLayerTable[i] = new float *[numHiddenNeurons];
    hiddenLayerNeuronTable[i] = new float[numHiddenNeurons];
    hiddenLayerNeuronBiasTable[i] = new float[numHiddenNeurons];

    for (uint16_t j = 0; j < numHiddenNeurons; j++)
    {
      hiddenLayerNeuronWeightLayerTable[i][j] = new float[numInputNeurons];
      hiddenLayerNeuronTable[i][j] = 0;
      hiddenLayerNeuronBiasTable[i][j] = ((float)random(-1000, 1000)) / 1000;

      for (uint16_t k = 0; k < numInputNeurons; k++)
      {
        hiddenLayerNeuronWeightLayerTable[i][j][k] =
            ((float)random(-1000, 1000)) / 1000;
      }
    }
  }

  network.numLayers = numHiddenLayers + 2;
  network.networkResponse = 0;

  // network.inputLayer.numRawInputs = numInputs;
  // network.inputLayer.rawInputs = rawInputArray;
  //
  // If initialising with this method, you must call NewInput()
  // with some inputs before you can use the network, to set these
  //
  network.inputLayer.numNeurons = numInputNeurons;
  network.inputLayer.neurons = new float[numInputNeurons];
  network.inputLayer.maxInput = maxInput;
  network.inputLayer.groupThresholds = new uint16_t[numInputNeurons];
  network.inputLayer.groupTotal = new uint16_t[numInputNeurons];

  network.outputLayer.numNeurons = numOutputNeurons;
  network.outputLayer.neurons = new float[numOutputNeurons];
  network.outputLayer.weightTable = outputLayerWeightTable;
  network.outputLayer.stringArray = outputArray;
  network.outputLayer.neuronBiasTable = outputLayerBiases;

  network.hiddenLayer.numNeurons = numHiddenNeurons;
  network.hiddenLayer.numLayers = numHiddenLayers;
  network.hiddenLayer.neuronTable = hiddenLayerNeuronTable;
  network.hiddenLayer.weightLayerTable = hiddenLayerNeuronWeightLayerTable;
  network.hiddenLayer.neuronBiasTable = hiddenLayerNeuronBiasTable;
}

void Ardbann::NewInput(uint16_t rawInputArray[], uint16_t numInputs)
{
  network.inputLayer.rawInputs = rawInputArray;
  network.inputLayer.numRawInputs = numInputs;
  CalculateInputNeurons();
}

void Ardbann::NewInput(Ardbann::SampleBuffer sampleBuffer, uint16_t numInputs)
{
  network.inputLayer.rawInputs = sampleBuffer.samples;
  network.inputLayer.numRawInputs = numInputs;
  CalculateInputNeurons();
}

void Ardbann::CalculateInputNeurons()
{
  uint8_t largestGroup = 0;

  for (uint16_t i = 0; i < network.inputLayer.numNeurons; i++)
  {
    network.inputLayer.groupThresholds[i] =
        ((network.inputLayer.maxInput + 1) / network.inputLayer.numNeurons) *
        (i + 1);
    network.inputLayer.groupTotal[i] = 0;
    // Serial.println(network.inputLayer.groupThresholds[i]);
  }

  for (uint16_t i = 0; i < network.inputLayer.numRawInputs; i++)
  {
    for (uint16_t j = 0; j < network.inputLayer.numNeurons; j++)
    {
      if (network.inputLayer.rawInputs[i] <=
          network.inputLayer.groupThresholds[j])
      {
        // Serial.printf("%d + 1 in group %d, ",
        // network.inputLayer.groupTotal[j],
        //              j);
        network.inputLayer.groupTotal[j] += 1;
        break;
      }
    }
  }

  for (uint16_t i = 0; i < network.inputLayer.numNeurons; i++)
  {
    if (network.inputLayer.groupTotal[i] >
        network.inputLayer.groupTotal[largestGroup])
    {
      largestGroup = i;
    }
  }

  for (uint16_t i = 0; i < network.inputLayer.numNeurons; i++)
  {
    // Serial.printf("group total i: %u group total largest: %u ",
    //              network.inputLayer.groupTotal[i],
    //              network.inputLayer.groupTotal[largestGroup]);
    network.inputLayer.neurons[i] = network.inputLayer.groupTotal[i] /
                                    network.inputLayer.groupTotal[largestGroup];
    // Serial.printf("input neuron %d = %.3f, ", i,
    // network.inputLayer.neurons[i]);
  }
}

uint8_t Ardbann::InputLayer()
{
  SumAndSquash(network.inputLayer.neurons, network.hiddenLayer.neuronTable[0],
               network.hiddenLayer.neuronBiasTable[0],
               network.hiddenLayer.weightLayerTable[0],
               network.inputLayer.numNeurons, network.hiddenLayer.numNeurons);
  // Serial.println("Done Input -> 1st Hidden Layer");
  for (uint8_t i = 1; i < network.hiddenLayer.numLayers; i++)
  {
    SumAndSquash(network.hiddenLayer.neuronTable[i - 1],
                 network.hiddenLayer.neuronTable[i],
                 network.hiddenLayer.neuronBiasTable[i],
                 network.hiddenLayer.weightLayerTable[i],
                 network.hiddenLayer.numNeurons,
                 network.hiddenLayer.numNeurons);
    // Serial.printf("Done Hidden Layer %d -> Hidden Layer %d\n", i - 1, i);
  }

  SumAndSquash(
      network.hiddenLayer.neuronTable[network.hiddenLayer.numLayers - 1],
      network.outputLayer.neurons, network.outputLayer.neuronBiasTable,
      network.outputLayer.weightTable, network.hiddenLayer.numNeurons,
      network.outputLayer.numNeurons);

  /*Serial.printf("Done Hidden Layer %d -> Output Layer\n",
                network.hiddenLayer.numLayers);*/

  network.networkResponse = OutputLayer();
  return network.networkResponse;
}

void Ardbann::SumAndSquash(float *Input, float *Output, float *Bias,
                           float **Weights, uint16_t numInputs,
                           uint16_t numOutputs)
{
  for (uint16_t i = 0; i < numOutputs; i++)
  {
    Output[i] = 0; // Bias[i];
    for (uint16_t j = 0; j < numInputs; j++)
    {
      Output[i] += Input[j] * Weights[i][j];
    }
    Output[i] = tanh(Output[i] * PI);

    // tanh is a quicker alternative to sigmoid
    // Serial.printf("i:%d This is the SumAndSquash Output %.2f\n", i,
    // Output[i]);
  }
}

uint8_t Ardbann::OutputLayer()
{
  uint8_t mostLikelyOutput = 0;

  for (uint16_t i = 0; i < network.outputLayer.numNeurons; i++)
  {
    if (network.outputLayer.neurons[i] >
        network.outputLayer.neurons[mostLikelyOutput])
    {
      mostLikelyOutput = i;
    }
    // Serial.printf("i: %d neuron: %-3f likely: %d\n", i,
    // network.outputLayer.neurons[i], mostLikelyOutput);
  }
  return mostLikelyOutput;
}

void Ardbann::PrintNetwork()
{
  Serial.print("\nInput: [");
  for (uint16_t i = 0; i < (network.inputLayer.numRawInputs - 1); i++)
  {
    Serial.print(network.inputLayer.rawInputs[i]);
    Serial.print(", ");
  }
  Serial.print(
      network.inputLayer.rawInputs[network.inputLayer.numRawInputs - 1]);
  Serial.print("]");

  Serial.printf("\nInput Layer | Hidden Layer ");

  if (network.hiddenLayer.numLayers > 1)
  {
    Serial.print("1 ");
    for (uint8_t i = 2; i <= network.hiddenLayer.numLayers; i++)
    {
      Serial.printf("| Hidden Layer %d ", i);
    }
  }

  Serial.println("| Output Layer");

  bool nothingLeft = false;
  uint16_t i = 0;
  while (nothingLeft == false)
  {
    if ((i >= network.inputLayer.numNeurons) &&
        (i >= network.hiddenLayer.numNeurons) &&
        (i >= network.outputLayer.numNeurons))
    {
      nothingLeft = true;
    }
    else
    {
      if (i < network.inputLayer.numNeurons)
      {
        Serial.printf("%-12.3f| ", network.inputLayer.neurons[i]);
      }
      else
      {
        Serial.print("            | ");
      }

      if (i < network.hiddenLayer.numNeurons)
      {
        if (network.hiddenLayer.numLayers == 1)
        {
          Serial.printf("%-13.3f| ", network.hiddenLayer.neuronTable[0][i]);
        }
        else
        {
          for (uint8_t j = 0; j < network.hiddenLayer.numLayers; j++)
          {
            Serial.printf("%-15.3f| ", network.hiddenLayer.neuronTable[j][i]);
          }
        }
      }
      else
      {
        Serial.print("             | ");
        if (network.hiddenLayer.numLayers > 1)
        {
          Serial.print("              | ");
        }
      }

      if (i < network.outputLayer.numNeurons)
      {
        Serial.printf("%.3f", network.outputLayer.neurons[i]);
      }
    }
    Serial.println();
    i++;
  }

  Serial.printf("I think this is output %d which is ", network.networkResponse);
  Serial.println(network.outputLayer.stringArray[network.networkResponse]);
}

void Ardbann::TrainDriver(float learningRate, bool verbose,
                          uint8_t numTrainingSets, uint8_t inputPin,
                          uint16_t bufferSize, long numSeconds)
{
  String serialInput;
  uint16_t trainingData[network.outputLayer.numNeurons][numTrainingSets]
                       [bufferSize],
      randomOutput, randomTrainingSet;

  for (uint8_t i = 0; i < network.outputLayer.numNeurons; i++)
  {
    Serial.printf("\nAttach the sensor to material %u: ", i);
    Serial.print(network.outputLayer.stringArray[i]);
    serialInput = Serial.readString();
    while (serialInput == NULL)
    {
      serialInput = Serial.readString();
    }
    Serial.println("\nReading...");
    for (uint8_t j = 0; j < numTrainingSets; j++)
    {
      Serial.printf("%u...", j + 1);
      for (uint16_t k = 0; k < bufferSize; k++)
      {
        trainingData[i][j][k] = analogRead(inputPin);
        Serial.printf("%u, ", trainingData[i][j][k]);
      }
      delay(50);
      Serial.println();
    }
  }

  if (verbose == true)
  {
    Serial.print("\nOutput Errors: \n");
  }

  unsigned long startTime = millis();
  numSeconds *= 1000;

  while ((millis() - startTime) < numSeconds)
  {
    randomOutput = random(0, network.outputLayer.numNeurons);
    randomTrainingSet = random(0, numTrainingSets);
    NewInput(trainingData[randomOutput][randomTrainingSet], bufferSize);
    InputLayer();

    if (verbose == true)
    {
      ErrorReporting(randomOutput);
      Serial.printf("%u | %u ", randomOutput, randomTrainingSet);
    }

    Train(randomOutput, learningRate);
  }
}

void Ardbann::TrainDriver(float learningRate, bool verbose,
                          uint8_t numTrainingSets, uint8_t inputPin,
                          uint16_t bufferSize, float desiredCost)
{
  String serialInput;
  uint16_t trainingData[network.outputLayer.numNeurons][numTrainingSets]
                       [bufferSize],
      randomOutput, randomTrainingSet;
  float currentCost[network.outputLayer.numNeurons] = {4.0};
  bool converged = false;

  for (uint8_t i = 0; i < network.outputLayer.numNeurons; i++)
  {
    Serial.printf("\nAttach the sensor to material %u: ", i);
    Serial.print(network.outputLayer.stringArray[i]);
    serialInput = Serial.readString();
    while (serialInput == NULL)
    {
      serialInput = Serial.readString();
    }
    Serial.println("Reading...");
    for (uint8_t j = 0; j < numTrainingSets; j++)
    {
      Serial.printf("%u...", j + 1);
      for (uint8_t k = 0; k < bufferSize; k++)
      {
        trainingData[i][j][k] = analogRead(inputPin);
        Serial.printf("%u, ", trainingData[i][j][k]);
      }
      delay(50);
      Serial.println();
    }
  }

  if (verbose == true)
  {
    Serial.print("\nOutput Errors: \n");
  }

  while (!converged)
  {
    randomOutput = random(0, network.outputLayer.numNeurons);
    randomTrainingSet = random(0, numTrainingSets);
    currentCost[randomOutput] = 0.0;
    NewInput(trainingData[randomOutput][randomTrainingSet], bufferSize);
    InputLayer();

    if (verbose == true)
    {
      ErrorReporting(randomOutput);
      Serial.printf("%u | %u ", randomOutput, randomTrainingSet);
    }

    Train(randomOutput, learningRate);
    for (uint8_t i = 0; i < network.outputLayer.numNeurons; i++)
    {
      if (i == randomOutput)
      {
        currentCost[randomOutput] += pow(1 - network.outputLayer.neurons[i], 2);
      }
      else
      {
        currentCost[randomOutput] += pow(network.outputLayer.neurons[i], 2);
      }
    }
    currentCost[randomOutput] /= network.outputLayer.numNeurons;

    for (uint8_t i = 0; i < network.outputLayer.numNeurons; i++)
    {
      Serial.print(currentCost[i]);
      Serial.print(", ");
      if (currentCost[i] > desiredCost)
      {
        break;
      }
      if (i == (network.outputLayer.numNeurons - 1))
      {
        converged = true;
      }
    }
    Serial.print(desiredCost);
  }
}

void Ardbann::Train(uint8_t correctOutput, float learningRate)
{
  float dOutputErrorToOutputSum[network.outputLayer.numNeurons] = {0.0};
  float dTotalErrorToHiddenNeuron = 0.0;
  float outputNeuronWeightChange[network.outputLayer.numNeurons]
                                [network.hiddenLayer.numNeurons] = {0.0};

  for (uint16_t i = 0; i < network.outputLayer.numNeurons; i++)
  {
    if (i == correctOutput)
    {
      dOutputErrorToOutputSum[i] =
          (1 - network.outputLayer.neurons[i]) *
          tanhDerivative(network.outputLayer.neurons[i]);
    }
    else
    {
      dOutputErrorToOutputSum[i] =
          -network.outputLayer.neurons[i] *
          tanhDerivative(network.outputLayer.neurons[i]);
    }
    // Serial.printf("\ndOutputErrorToOutputSum[%d]: %.3f", i,
    // dOutputErrorToOutputSum[i]);
    for (uint16_t j = 0; j < network.hiddenLayer.numNeurons; j++)
    {
      outputNeuronWeightChange[i][j] =
          dOutputErrorToOutputSum[i] *
          network.hiddenLayer
              .neuronTable[network.hiddenLayer.numLayers - 1][j] *
          learningRate;
      // Serial.printf("\n  outputNeuronWeightChange[%d][%d]: %.3f", i, j,
      //              outputNeuronWeightChange[i][j]);
    }
  }

  for (uint16_t i = 0; i < network.hiddenLayer.numNeurons; i++)
  {
    dTotalErrorToHiddenNeuron = 0.0;
    for (uint16_t j = 0; j < network.outputLayer.numNeurons; j++)
    {
      dTotalErrorToHiddenNeuron +=
          dOutputErrorToOutputSum[j] * network.outputLayer.weightTable[j][i];
      // Serial.printf("\nOld Output Weight[%d][%d]: %.3f", i, j,
      // network.outputLayer.weightTable[j][i]);
      network.outputLayer.weightTable[j][i] += outputNeuronWeightChange[j][i];
      // Serial.printf("\nNew Output Weight[%d][%d]: %.3f", i, j,
      // network.outputLayer.weightTable[j][i]);
    }
    for (uint16_t k = 0; k < network.inputLayer.numNeurons; k++)
    {
      // Serial.printf("\nOld Hidden Weight[%d][%d]: %.3f", i, k,
      // network.hiddenLayer.weightLayerTable[0][i][k]);
      network.hiddenLayer.weightLayerTable[0][i][k] +=
          dTotalErrorToHiddenNeuron *
          tanhDerivative(network.hiddenLayer.neuronTable[0][i]) *
          network.inputLayer.neurons[k] * learningRate;
      // Serial.printf("\nNew Hidden Weight[%d][%d]: %.3f", i, k,
      // network.hiddenLayer.weightLayerTable[0][i][k]);
    }
  }
}

float Ardbann::tanhDerivative(float inputValue)
{
  // if (inputValue < 0)
  //{
  //  return -1 * (1 - pow(tanh(inputValue), 2));
  //}
  // else
  //{
  return 1 - pow(tanh(inputValue * PI), 2);
  //}
}

void Ardbann::PrintInputNeuronDetails(uint8_t neuronNum)
{
  if (neuronNum < network.inputLayer.numNeurons)
  {
    Serial.printf("\nInput Neuron %d: %.3f\n", neuronNum,
                  network.inputLayer.neurons[neuronNum]);
  }
  else
  {
    Serial.printf(
        "\nERROR: You've asked for input neuron %d when only %d exist\n",
        neuronNum, network.inputLayer.numNeurons);
  }
}

void Ardbann::PrintOutputNeuronDetails(uint8_t neuronNum)
{
  if (neuronNum < network.outputLayer.numNeurons)
  {

    Serial.printf("\nOutput Neuron %d:\n", neuronNum);

    for (uint16_t i = 0; i < network.hiddenLayer.numNeurons; i++)
    {
      Serial.printf(
          "%.3f-*->%.3f |",
          network.hiddenLayer.neuronTable[network.hiddenLayer.numLayers - 1][i],
          network.outputLayer.weightTable[neuronNum][i]);

      if (i == floor(network.hiddenLayer.numNeurons / 2))
      {
        Serial.printf(" = %.3f", network.outputLayer.neurons[neuronNum]);
      }
      Serial.println();
    }
  }
  else
  {
    Serial.printf(
        "\nERROR: You've asked for output neuron %d when only %d exist\n",
        neuronNum, network.outputLayer.numNeurons);
  }
}

void Ardbann::PrintHiddenNeuronDetails(uint8_t layerNum, uint8_t neuronNum)
{
  if (neuronNum < network.hiddenLayer.numNeurons)
  {

    Serial.printf("\nHidden Neuron %d:\n", neuronNum);

    if (layerNum == 0)
    {

      for (uint16_t i = 0; i < network.inputLayer.numNeurons; i++)
      {
        Serial.printf("%.3f-*->%.3f |", network.inputLayer.neurons[i],
                      network.hiddenLayer.weightLayerTable[0][neuronNum][i]);

        if (i == floor(network.inputLayer.numNeurons / 2))
        {
          Serial.printf(" = %.3f",
                        network.hiddenLayer.neuronTable[0][neuronNum]);
        }
        Serial.println();
      }
    }
    else
    {

      for (uint16_t i = 0; i < network.hiddenLayer.numNeurons; i++)
      {
        Serial.printf(
            "%.3f-*->%.3f |", network.hiddenLayer.neuronTable[layerNum - 1][i],
            network.hiddenLayer.weightLayerTable[layerNum][neuronNum][i]);

        if (i == floor(network.hiddenLayer.numNeurons / 2))
        {
          Serial.printf(" = %.3f",
                        network.hiddenLayer.neuronTable[0][neuronNum]);
        }
        Serial.println();
      }
    }
  }
  else
  {
    Serial.printf(
        "\nERROR: You've asked for hidden neuron %d when only %d exist\n",
        neuronNum, network.hiddenLayer.numNeurons);
  }
}

void Ardbann::ErrorReporting(uint8_t correctResponse)
{
  Serial.println();
  for (uint8_t i = 0; i < network.outputLayer.numNeurons; i++)
  {
    if (i == correctResponse)
    {
      Serial.printf("%-7.3f | ",
                    (1 - network.outputLayer.neurons[correctResponse]));
    }
    else
    {
      Serial.printf("%-7.3f | ", -network.outputLayer.neurons[i]);
    }
  }
}