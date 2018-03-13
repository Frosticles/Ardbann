/*
  Ardbann.h - ARDuino Backpropogating Artificial Neural Network.
  Created by Peter Frost, February 9, 2017.
  Released into the public domain.
*/

#include "Ardbann.h"

Ardbann::Ardbann(uint16_t rawInputArray[], const uint16_t maxInput,
                 String outputArray[], const uint16_t numInputs,
                 const uint16_t numInputNeurons,
                 const uint16_t numHiddenNeurons, const uint8_t numHiddenLayers,
                 const uint16_t numOutputNeurons) {

  float **outputLayerWeightTable = new float *[numOutputNeurons];

  for (uint8_t i = 0; i < numOutputNeurons; i++) {
    outputLayerWeightTable[i] = new float[numHiddenNeurons];
    for (uint16_t j = 0; j < numHiddenNeurons; j++) {
      outputLayerWeightTable[i][j] = 0.0;
    }
  }

  float ***hiddenLayerWeightLayerTable = new float **[numHiddenLayers];
  float **hiddenLayerNeuronTable = new float *[numHiddenLayers];

  for (uint8_t i = 0; i < numHiddenLayers; i++) {
    hiddenLayerWeightLayerTable[i] = new float *[numHiddenNeurons];
    hiddenLayerNeuronTable[i] = new float[numHiddenNeurons];

    for (uint16_t j = 0; j < numHiddenNeurons; j++) {
      hiddenLayerWeightLayerTable[i][j] = new float[numInputNeurons];
      hiddenLayerNeuronTable[i][j] = 0;

      for (uint16_t k = 0; k < numInputNeurons; k++) {
        hiddenLayerWeightLayerTable[i][j][k] = 0.0;
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
  network.inputLayer.groupTotal = new double[numInputNeurons];

  CalculateInputNeurons();

  network.outputLayer.numNeurons = numOutputNeurons;
  network.outputLayer.neurons = new float[numOutputNeurons];
  network.outputLayer.weightTable = outputLayerWeightTable;
  network.outputLayer.stringArray = outputArray;

  network.hiddenLayer.numNeurons = numHiddenNeurons;
  network.hiddenLayer.numLayers = numHiddenLayers;
  network.hiddenLayer.neuronTable = hiddenLayerNeuronTable;
  network.hiddenLayer.weightLayerTable = hiddenLayerWeightLayerTable;
}

void Ardbann::NewInput(uint16_t rawInputArray[], uint32_t numInputs) {
  network.inputLayer.rawInputs = rawInputArray;
  network.inputLayer.numRawInputs = numInputs;
  CalculateInputNeurons();
}

void Ardbann::NewInput(Ardbann::SampleBuffer sampleBuffer, uint32_t numInputs) {
  network.inputLayer.rawInputs = sampleBuffer.samples;
  network.inputLayer.numRawInputs = numInputs;
  CalculateInputNeurons();
}

void Ardbann::CalculateInputNeurons() {
  uint8_t largestGroup = 0;

  for (uint16_t i = 0; i < network.inputLayer.numNeurons; i++) {
    network.inputLayer.groupThresholds[i] =
        (network.inputLayer.maxInput / network.inputLayer.numNeurons) * (i + 1);
    network.inputLayer.groupTotal[i] = 0.0;
  }

  for (uint16_t i = 0; i < network.inputLayer.numRawInputs; i++) {
    for (uint16_t j = 0; j < network.inputLayer.numNeurons; j++) {
      if (network.inputLayer.rawInputs[i] <
          network.inputLayer.groupThresholds[j]) {
        // Serial.printf("%.1f + %d, in group %d\n", groupTotal[j],
        // network.inputLayer.rawInputs[i], j);
        network.inputLayer.groupTotal[j] += network.inputLayer.rawInputs[i];
        break;
      }
    }
  }

  for (uint16_t i = 0; i < network.inputLayer.numRawInputs; i++) {
    if (network.inputLayer.groupTotal[i] >
        network.inputLayer.groupTotal[largestGroup]) {
      largestGroup = i;
    }
  }

  for (uint16_t i = 0; i < network.inputLayer.numNeurons; i++) {
    network.inputLayer.neurons[i] = network.inputLayer.groupTotal[i] /
                                    network.inputLayer.groupTotal[largestGroup];
    // Serial.println(network.inputLayer.neurons[i]);
  }
}

uint8_t Ardbann::InputLayer() {
  SumAndSquash(network.inputLayer.neurons, network.hiddenLayer.neuronTable[0],
               network.hiddenLayer.weightLayerTable[0],
               network.inputLayer.numNeurons, network.hiddenLayer.numNeurons);
  // Serial.println("Done Input -> 1st Hidden Layer");
  for (uint8_t i = 1; i < network.hiddenLayer.numLayers; i++) {
    SumAndSquash(network.hiddenLayer.neuronTable[i - 1],
                 network.hiddenLayer.neuronTable[i],
                 network.hiddenLayer.weightLayerTable[i],
                 network.hiddenLayer.numNeurons,
                 network.hiddenLayer.numNeurons);
    // Serial.printf("Done Hidden Layer %d -> Hidden Layer %d\n", i - 1, i);
  }

  SumAndSquash(
      network.hiddenLayer.neuronTable[network.hiddenLayer.numLayers - 1],
      network.outputLayer.neurons, network.outputLayer.weightTable,
      network.hiddenLayer.numNeurons, network.outputLayer.numNeurons);

  /*Serial.printf("Done Hidden Layer %d -> Output Layer\n",
                network.hiddenLayer.numLayers);*/

  network.networkResponse = OutputLayer();
  return network.networkResponse;
}

void Ardbann::SumAndSquash(float *pointerToInput, float *pointerToOutput,
                           float **pointerToWeights, uint16_t numInputs,
                           uint16_t numOutputs) {
  for (uint16_t i = 0; i < numOutputs; i++) {
    pointerToOutput[i] = 0;
    for (uint16_t j = 0; j < numInputs; j++) {
      pointerToOutput[i] += pointerToInput[j] * pointerToWeights[i][j];
    }
    pointerToOutput[i] = tanh(pointerToOutput[i] *
                              PI); // tanh is a quicker alternative to sigmoid
    // Serial.printf("i:%d This is the SumAndSquash Output %.2f\n", i,
    // pointerToOutput[i]);
  }
}

uint8_t Ardbann::OutputLayer() {
  uint8_t mostLikelyOutput = 0;

  for (uint16_t i = 0; i < network.outputLayer.numNeurons; i++) {
    if (network.outputLayer.neurons[i] >
        network.outputLayer.neurons[mostLikelyOutput]) {
      mostLikelyOutput = i;
    }
    // Serial.printf("i: %d neuron: %-3f likely: %d\n", i,
    // network.outputLayer.neurons[i], mostLikelyOutput);
  }
  return mostLikelyOutput;
}

void Ardbann::PrintNetwork() {
  Serial.print("\nInput: [");
  for (uint16_t i = 0; i < (network.inputLayer.numRawInputs - 1); i++) {
    Serial.printf("%d, ", network.inputLayer.rawInputs[i]);
  }
  Serial.printf(
      "%d]\n",
      network.inputLayer.rawInputs[network.inputLayer.numRawInputs - 1]);

  Serial.printf("\nInput Layer | Hidden Layer ");

  if (network.hiddenLayer.numLayers > 1) {
    Serial.print("1 ");
    for (uint8_t i = 2; i <= network.hiddenLayer.numLayers; i++) {
      Serial.printf("| Hidden Layer %d ", i);
    }
  }

  Serial.println("| Output Layer");

  bool nothingLeft = false;
  uint16_t i = 0;
  while (nothingLeft == false) {
    if ((i >= network.inputLayer.numNeurons) &&
        (i >= network.hiddenLayer.numNeurons) &&
        (i >= network.outputLayer.numNeurons)) {
      nothingLeft = true;
    } else {
      if (i < network.inputLayer.numNeurons) {
        Serial.printf("%-12.3f| ", network.inputLayer.neurons[i]);
      } else {
        Serial.print("            | ");
      }

      if (i < network.hiddenLayer.numNeurons) {
        if (network.hiddenLayer.numLayers == 1) {
          Serial.printf("%-13.3f| ", network.hiddenLayer.neuronTable[0][i]);
        } else {
          for (uint8_t j = 0; j < network.hiddenLayer.numLayers; j++) {
            Serial.printf("%-15.3f| ", network.hiddenLayer.neuronTable[j][i]);
          }
        }
      } else {
        Serial.print("             | ");
        if (network.hiddenLayer.numLayers > 1) {
          Serial.print("              | ");
        }
      }

      if (i < network.outputLayer.numNeurons) {
        Serial.printf("%.3f", network.outputLayer.neurons[i]);
      }
    }
    Serial.println();
    i++;
  }

  Serial.printf("I think this is output %d which is ", network.networkResponse);
  Serial.println(network.outputLayer.stringArray[network.networkResponse]);
}

void Ardbann::Train(uint8_t correctOutput, uint32_t numSeconds,
                    float learningRate, bool showError) {

  uint8_t networkResponse = InputLayer();
  uint32_t startingMillis = millis();
  numSeconds = numSeconds * 1000;

  while ((millis() - startingMillis) < numSeconds) {

    if (showError == true) {
      ErrorReporting(correctOutput);
    }

    for (uint16_t i = 0; i < network.outputLayer.numNeurons; i++) {
      for (uint16_t j = 0; j < network.hiddenLayer.numNeurons; j++) {
        if (i == correctOutput) {
          if (j == correctOutput) {
            network.outputLayer.weightTable[i][j] +=
                learningRate *
                tanhDerivative(
                    1 -
                    (network.hiddenLayer
                         .neuronTable[network.hiddenLayer.numLayers - 1][j] *
                     network.outputLayer.weightTable[i][j])) *
                pow(1 - (network.hiddenLayer
                             .neuronTable[network.hiddenLayer.numLayers - 1]
                                         [j] *
                         network.outputLayer.weightTable[i][j]),
                    2);
          } else {
            network.outputLayer.weightTable[i][j] +=
                learningRate *
                tanhDerivative(
                    0 -
                    (network.hiddenLayer
                         .neuronTable[network.hiddenLayer.numLayers - 1][j] *
                     network.outputLayer.weightTable[i][j])) *
                pow(0 - (network.hiddenLayer
                             .neuronTable[network.hiddenLayer.numLayers - 1]
                                         [j] *
                         network.outputLayer.weightTable[i][j]),
                    2);
          }
        } else {
          // Negative Training
        }
      }
    }

    for (uint8_t i = network.hiddenLayer.numLayers - 1; i > 0; i--) {
      for (uint16_t j = 0; j < network.hiddenLayer.numNeurons; j++) {
        for (uint16_t k = 0; k < network.inputLayer.numNeurons; k++) {
          if (j == correctOutput) {
            if (k == correctOutput) {
              network.hiddenLayer.weightLayerTable[i][j][k] +=
                  learningRate *
                  tanhDerivative(
                      1 - (network.hiddenLayer.neuronTable[i][j] *
                           network.hiddenLayer.weightLayerTable[i][j][k])) *
                  pow(1 - (network.hiddenLayer.neuronTable[i][j] *
                           network.hiddenLayer.weightLayerTable[i][j][k]),
                      2);
            } else {
              network.hiddenLayer.weightLayerTable[i][j][k] +=
                  learningRate *
                  tanhDerivative(
                      0 - (network.hiddenLayer.neuronTable[i][j] *
                           network.hiddenLayer.weightLayerTable[i][j][k])) *
                  pow(0 - (network.hiddenLayer.neuronTable[i][j] *
                           network.hiddenLayer.weightLayerTable[i][j][k]),
                      2);
            }
          } else {
            // Negative Training
          }
        }
      }
    }

    for (uint16_t i = 0; i < network.hiddenLayer.numNeurons; i++) {
      for (uint16_t j = 0; j < network.inputLayer.numNeurons; j++) {

        if (i == correctOutput) {
          if (j == correctOutput) {

            network.hiddenLayer.weightLayerTable[0][i][j] +=
                learningRate *
                tanhDerivative(
                    1 - (network.inputLayer.neurons[j] *
                         network.hiddenLayer.weightLayerTable[0][i][j])) *
                pow(1 - (network.inputLayer.neurons[j] *
                         network.hiddenLayer.weightLayerTable[0][i][j]),
                    2);

          } else {

            network.hiddenLayer.weightLayerTable[0][i][j] +=
                learningRate *
                tanhDerivative(
                    0 - (network.inputLayer.neurons[j] *
                         network.hiddenLayer.weightLayerTable[0][i][j])) *
                pow(0 - (network.inputLayer.neurons[j] *
                         network.hiddenLayer.weightLayerTable[0][i][j]),
                    2);
          }
        } else {
          /*derivativeRate[i][j] =
          tanhDerivative(-1 - (network.inputLayer.neurons[j] *
          network.hiddenLayer.weightLayerTable[0][i][j]))
          * pow(-1 - (network.inputLayer.neurons[j] *
          network.hiddenLayer.weightLayerTable[0][i][j]), 2);
          network.hiddenLayer.weightLayerTable[0][i][j] += derivativeRate[i][j]
          * learningRate;*/
        }
      }
    }
    networkResponse = InputLayer();
  }
}

float Ardbann::tanhDerivative(float inputValue) {
  if (inputValue < 0) {
    return -1 * (1 - pow(tanh(inputValue), 2));
  } else {
    return 1 - pow(tanh(inputValue), 2);
  }
}

void Ardbann::PrintInputNeuronDetails(uint8_t neuronNum) {
  if (neuronNum < network.inputLayer.numNeurons) {
    Serial.printf("\nInput Neuron %d: %.3f\n", neuronNum,
                  network.inputLayer.neurons[neuronNum]);
  } else {
    Serial.printf(
        "\nERROR: You've asked for input neuron %d when only %d exist\n",
        neuronNum, network.inputLayer.numNeurons);
  }
}

void Ardbann::PrintOutputNeuronDetails(uint8_t neuronNum) {
  if (neuronNum < network.outputLayer.numNeurons) {

    Serial.printf("\nOutput Neuron %d:\n", neuronNum);

    for (uint16_t i = 0; i < network.hiddenLayer.numNeurons; i++) {
      Serial.printf(
          "%.3f---%.3f |",
          network.hiddenLayer.neuronTable[network.hiddenLayer.numLayers - 1][i],
          network.outputLayer.weightTable[neuronNum][i]);

      if (i == floor(network.hiddenLayer.numNeurons / 2)) {
        Serial.printf(" = %.3f", network.outputLayer.neurons[neuronNum]);
      }
      Serial.println();
    }
  } else {
    Serial.printf(
        "\nERROR: You've asked for output neuron %d when only %d exist\n",
        neuronNum, network.outputLayer.numNeurons);
  }
}

void Ardbann::PrintHiddenNeuronDetails(uint8_t layerNum, uint8_t neuronNum) {
  if (neuronNum < network.hiddenLayer.numNeurons) {

    Serial.printf("\nHidden Neuron %d:\n", neuronNum);

    if (layerNum == 0) {

      for (uint16_t i = 0; i < network.inputLayer.numNeurons; i++) {
        Serial.printf("%.3f---%.3f |", network.inputLayer.neurons[i],
                      network.hiddenLayer.weightLayerTable[0][neuronNum][i]);

        if (i == floor(network.inputLayer.numNeurons / 2)) {
          Serial.printf(" = %.3f",
                        network.hiddenLayer.neuronTable[0][neuronNum]);
        }
        Serial.println();
      }

    } else {

      for (uint16_t i = 0; i < network.hiddenLayer.numNeurons; i++) {
        Serial.printf(
            "%.3f---%.3f |", network.hiddenLayer.neuronTable[layerNum - 1][i],
            network.hiddenLayer.weightLayerTable[layerNum][neuronNum][i]);

        if (i == floor(network.hiddenLayer.numNeurons / 2)) {
          Serial.printf(" = %.3f",
                        network.hiddenLayer.neuronTable[0][neuronNum]);
        }
        Serial.println();
      }
    }
  } else {
    Serial.printf(
        "\nERROR: You've asked for hidden neuron %d when only %d exist\n",
        neuronNum, network.hiddenLayer.numNeurons);
  }
}

void Ardbann::ErrorReporting(uint16_t correctResponse) {
  Serial.printf("\nError Report\nInput Layer | Hidden Layer ");

  if (network.hiddenLayer.numLayers > 1) {
    Serial.print("1 ");
    for (uint8_t i = 2; i <= network.hiddenLayer.numLayers; i++) {
      Serial.printf("| Hidden Layer %d ", i);
    }
  }
  Serial.println("| Output Layer");

  Serial.printf("%-12.3f| ", 1 - network.inputLayer.neurons[correctResponse]);

  for (uint8_t i = 0; i < network.numLayers - 2; i++) {
    Serial.printf("%-15.3f| ",
                  1 - network.hiddenLayer.neuronTable[i][correctResponse]);
  }

  Serial.printf("%.3f\n", 1 - network.outputLayer.neurons[correctResponse]);
}