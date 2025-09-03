#include <Arduino.h>


// include main library header file
#include <TensorFlowLite.h>

// include static array definition of pre-trained model
#include "model.h"


// This TensorFlow Lite Micro Library for Arduino is not similar to standard
// Arduino libraries. These additional header files must be included.
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Globals pointers, used to address TensorFlow Lite components.
// Pointers are not usual in Arduino sketches, future versions of
// the library may change this...
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// There is no way to calculate this parameter
// the value is usually determined by trial and errors
// It is the dimension of the memory area used by the TFLite interpreter
// to store tensors and intermediate results
constexpr int kTensorArenaSize = 2000;

// Keep aligned to 16 bytes for CMSIS (Cortex Microcontroller Software Interface Standard)
// alignas(16) directive is used to specify that the array 
// should be stored in memory at an address that is a multiple of 16.
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

unsigned long boot_time = 0;
unsigned long last_status = 0;
bool loading = true;

void setup() {
  Serial.begin(115200);
  while(!Serial);

  Serial.println("Sine(x) function inference example.");
  Serial.println("Initializing TensorFlow Lite Micro Interpreter...");

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);

  // Check if model and library have compatible schema version,
  // if not, there is a misalignement between TensorFlow version used
  // to train and generate the TFLite model and the current version of library
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model provided and schema version are not equal!");
    while(true); // stop program here
  }

  // This pulls in all the TensorFlow Lite operators.
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  // if an error occurs, stop the program.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while(true); // stop program here
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Initialization done.");
  Serial.println("");
  Serial.println("Please, input a float number between 0 and 6.28");

  boot_time = millis();
}

void printLoading() {
    Serial.println("======================================");
    Serial.println("ESP32-S3 Touch AMOLED 1.75-B Hardware Test");
    Serial.println("======================================\n");
    
    // Check PSRAM
    Serial.println("=== Memory Check ===");
    if (psramFound()) {
        float psramSize = ESP.getPsramSize();
        float freePsram = ESP.getFreePsram();
        if (psramSize >= 1024 * 1024) {
            Serial.printf("PSRAM found: %.2f MB\r\n", psramSize / (1024.0 * 1024.0));
        } else {
            Serial.printf("PSRAM found: %.2f KB\r\n", psramSize / 1024.0);
        }
        if (freePsram >= 1024 * 1024) {
            Serial.printf("Free PSRAM: %.2f MB\r\n", freePsram / (1024.0 * 1024.0));
        } else {
            Serial.printf("Free PSRAM: %.2f KB\r\n", freePsram / 1024.0);
        }
    } else {
        Serial.println("PSRAM not found!\n");
    }
    float freeHeap = ESP.getFreeHeap();
    if (freeHeap >= 1024 * 1024) {
        Serial.printf("Free heap: %.2f MB\r\n", freeHeap / (1024.0 * 1024.0));
    } else {
        Serial.printf("Free heap: %.2f KB\r\n", freeHeap / 1024.0);
    }
    Serial.printf("CPU frequency: %d MHz\r\n", ESP.getCpuFreqMHz());
    Serial.println("=== End Memory Check ===\n");
}

void checkMemory() {
    unsigned long uptime = millis() - boot_time;

    // Calculate memory usage
    uint32_t total_heap = ESP.getHeapSize();
    uint32_t free_heap = ESP.getFreeHeap();
    uint32_t used_heap = total_heap - free_heap;
    float ram_usage = (float)used_heap / total_heap * 100.0;

    // Calculate flash usage
    uint32_t sketch_size = ESP.getSketchSize();
    uint32_t total_flash = ESP.getFlashChipSize();
    float flash_usage = (float)sketch_size / total_flash * 100.0;

    Serial.println("=== System Status ===");
    Serial.printf("Uptime: %lu ms (%.1f seconds)\r\n", uptime, uptime / 1000.0);

    // RAM Usage
    if (total_heap >= 1024 * 1024) {
        Serial.printf("RAM Usage: %.1f%% (%.2f MB / %.2f MB)\r\n", ram_usage, used_heap / (1024.0 * 1024.0), total_heap / (1024.0 * 1024.0));
    } else {
        Serial.printf("RAM Usage: %.1f%% (%.2f KB / %.2f KB)\r\n", ram_usage, used_heap / 1024.0, total_heap / 1024.0);
    }

    // Flash Usage
    if (total_flash >= 1024 * 1024) {
        Serial.printf("Flash Usage: %.1f%% (%.2f MB / %.2f MB)\r\n", flash_usage, sketch_size / (1024.0 * 1024.0), total_flash / (1024.0 * 1024.0));
    } else {
        Serial.printf("Flash Usage: %.1f%% (%.2f KB / %.2f KB)\r\n", flash_usage, sketch_size / 1024.0, total_flash / 1024.0);
    }

    // Free PSRAM
    uint32_t free_psram = ESP.getFreePsram();
    if (free_psram >= 1024 * 1024) {
        Serial.printf("Free PSRAM: %.2f MB\r\n", free_psram / (1024.0 * 1024.0));
    } else {
        Serial.printf("Free PSRAM: %.2f KB\r\n", free_psram / 1024.0);
    }
    Serial.println("=== End Status ===\n"); 
}

void loop() {
    if (loading) {
        Serial.println("Loading...");
        delay(5000);
        printLoading();
        Serial.println("Loading complete");
        loading = false;
    }

    // Print status every 10 seconds
    if (millis() - last_status > 30000) {
        checkMemory();
        last_status = millis();
    }

    //delay(1000); // 1 second delay
    //Serial.printf("Boot time: %d seconds\r\n", (millis() - boot_time) / 1000);

  // Check if a value was sent from Serial Monitor
  // if so, 'sanitize' the input and perform inference
  if (Serial.available()){
    String inputValue = Serial.readStringUntil('\n');
    float x = inputValue.toFloat(); // evaluates to zero if the user input is not a valid number
    Serial.print("Your input value: ");
    Serial.println(x);
    // The model was trained in range 0 to 2*Pi
    // if the value provided by user is not in this range
    // the value is corrected substituting edge values
    if (x<0) x = 0;
    if (x >6.28) x = 6.28;
    Serial.print("Adapted input value: ");
    Serial.println(x);
  
    // Quantize the input from floating-point to integer
    // because model has been optimized by quantization
    int8_t x_quantized = x / input->params.scale + input->params.zero_point;

    // Place the quantized input in the model's input tensor
    input->data.int8[0] = x_quantized;

    // Run inference, and report if an error occurs
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      Serial.println("Invoke failed!");
      return;
    }

    // Obtain the quantized output from model's output tensor
    int8_t y_quantized = output->data.int8[0];
    // Dequantize the output from integer to floating-point
    float y = (y_quantized - output->params.zero_point) * output->params.scale;

    Serial.print("Inferred Sin of ");
    Serial.print(x);
    Serial.print(" = ");
    Serial.println(y,2);
    Serial.print("Actual Sin of ");
    Serial.print(x);
    Serial.print(" = ");
    Serial.println(sin(x),2);

  }
}
