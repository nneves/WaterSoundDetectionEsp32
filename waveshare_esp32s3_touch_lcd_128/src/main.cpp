#include <Arduino.h>
#include <math.h>

// include main library header file
#include <TensorFlowLite.h>

// include static array definition of pre-trained model
#include "model_improved_int8_3_2_1.h"


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
// Increased size for 70KB water prediction model with MFCC features
// Model size + intermediate tensors + working memory
constexpr int kTensorArenaSize = 58 * 1024;  // 140KB for model operations

// Keep aligned to 16 bytes for CMSIS (Cortex Microcontroller Software Interface Standard)
// alignas(16) directive is used to specify that the array 
// should be stored in memory at an address that is a multiple of 16.
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

unsigned long boot_time = 0;
unsigned long last_status = 0;
bool loading = true;

// Audio processing configuration (matching Python script)
const int CHUNK_DURATION = 3;  // seconds
const int SAMPLE_RATE = 16000;
const int SAMPLES_PER_CHUNK = CHUNK_DURATION * SAMPLE_RATE;
const int N_MFCC = 13;
const int MAX_SEQUENCE_LENGTH = 200;

// MFCC processing buffers
float audio_buffer[SAMPLES_PER_CHUNK];
float mfcc_buffer[N_MFCC * MAX_SEQUENCE_LENGTH];
int8_t quantized_input[N_MFCC * MAX_SEQUENCE_LENGTH];

// Function declarations
void extractMFCCFeatures(float* audio_data, int audio_length, float* mfcc_output);
void normalizeMFCC(float* mfcc_data, int length);
void quantizeMFCC(float* mfcc_data, int8_t* quantized_data, int length, 
                  float input_scale, int input_zero_point);
void generateMockAudio(float* audio_data, int length);
void performWaterPrediction();
void testWithMockAudio();
void checkMemoryUsage();
void showHelp();
void printMFCCStats(float* mfcc_data, int length, const char* label);

// Enhanced MFCC-like feature extraction with better frequency analysis
void extractMFCCFeatures(float* audio_data, int audio_length, float* mfcc_output) {
  // Clear output buffer
  for (int i = 0; i < N_MFCC * MAX_SEQUENCE_LENGTH; i++) {
    mfcc_output[i] = 0.0f;
  }
  
  // Calculate sequence length (simplified)
  int sequence_length = min(audio_length / 80, MAX_SEQUENCE_LENGTH); // ~80 samples per frame
  
  // Enhanced feature extraction with better frequency analysis
  for (int t = 0; t < sequence_length; t++) {
    int start_sample = t * 80;
    int end_sample = min(start_sample + 80, audio_length);
    
    // Calculate frame energy
    float frame_energy = 0.0f;
    for (int s = start_sample; s < end_sample; s++) {
      frame_energy += audio_data[s] * audio_data[s];
    }
    frame_energy = sqrt(frame_energy / (end_sample - start_sample));
    
    // Calculate frequency domain features (simplified)
    float freq_bands[N_MFCC];
    
    // Band 0: DC component (average)
    freq_bands[0] = 0.0f;
    for (int s = start_sample; s < end_sample; s++) {
      freq_bands[0] += audio_data[s];
    }
    freq_bands[0] /= (end_sample - start_sample);
    
    // Bands 1-12: Different frequency ranges
    for (int m = 1; m < N_MFCC; m++) {
      float freq_sum = 0.0f;
      float freq_weight = (float)m / N_MFCC; // 0 to 1
      
      // Simulate frequency analysis with different frequency components
      for (int s = start_sample; s < end_sample; s++) {
        float t_norm = (float)(s - start_sample) / (end_sample - start_sample);
        float freq = 50.0f + freq_weight * 2000.0f; // 50Hz to 2050Hz range
        freq_sum += audio_data[s] * sin(2.0f * PI * freq * t_norm);
      }
      freq_bands[m] = freq_sum / (end_sample - start_sample);
    }
    
    // Apply mel-scale-like transformation and create MFCC-like features
    for (int m = 0; m < N_MFCC; m++) {
      int idx = t * N_MFCC + m;
      
      // Apply mel-scale-like weighting
      float mel_weight = 1.0f;
      if (m > 0) {
        mel_weight = log(1.0f + (float)m * 0.1f); // Approximate mel scale
      }
      
      // Combine energy and frequency information
      float mfcc_value = freq_bands[m] * mel_weight * frame_energy;
      
      // Apply DCT-like transformation (simplified)
      if (m > 0) {
        mfcc_value *= cos(PI * m * t / sequence_length);
      }
      
      mfcc_output[idx] = mfcc_value;
    }
  }
}

// Normalize MFCC features with fixed scaling to preserve differences
void normalizeMFCC(float* mfcc_data, int length) {
  // Use a fixed scaling factor instead of normalizing to max
  // This preserves the relative differences between different audio patterns
  float scale_factor = 10.0f; // Adjust this to control the range
  
  for (int i = 0; i < length; i++) {
    mfcc_data[i] = mfcc_data[i] * scale_factor;
    
    // Clamp to reasonable range to avoid extreme values
    if (mfcc_data[i] > 1.0f) mfcc_data[i] = 1.0f;
    if (mfcc_data[i] < -1.0f) mfcc_data[i] = -1.0f;
  }
}

// Quantize MFCC features for INT8 model
void quantizeMFCC(float* mfcc_data, int8_t* quantized_data, int length, 
                  float input_scale, int input_zero_point) {
  for (int i = 0; i < length; i++) {
    float quantized = mfcc_data[i] / input_scale + input_zero_point;
    quantized_data[i] = (int8_t)max(-128, min(127, (int)round(quantized)));
  }
}

// Generate mock audio data for testing (replace with real microphone input)
void generateMockAudio(float* audio_data, int length) {
  // Generate a simple sine wave with some noise to simulate water flow
  for (int i = 0; i < length; i++) {
    float t = (float)i / SAMPLE_RATE;
    // Simulate water flow sound with multiple frequencies
    audio_data[i] = 0.3f * sin(2.0f * PI * 100.0f * t) +  // Low frequency
                    0.2f * sin(2.0f * PI * 500.0f * t) +  // Mid frequency
                    0.1f * sin(2.0f * PI * 2000.0f * t) + // High frequency
                    0.05f * ((float)random(-1000, 1000) / 1000.0f); // Noise
  }
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

void performWaterPrediction() {
    Serial.println("=== Water Consumption Prediction ===");
    
    // Check memory before processing
    Serial.printf("Free heap before processing: %d bytes\r\n", ESP.getFreeHeap());
    
    // Generate mock audio data (replace with real microphone input)
    Serial.println("Generating mock audio data...");
    generateMockAudio(audio_buffer, SAMPLES_PER_CHUNK);
    
    // Extract MFCC features
    Serial.println("Extracting MFCC features...");
    extractMFCCFeatures(audio_buffer, SAMPLES_PER_CHUNK, mfcc_buffer);
    
    // Normalize features
    normalizeMFCC(mfcc_buffer, N_MFCC * MAX_SEQUENCE_LENGTH);
    
    // Quantize features for INT8 model
    float input_scale = input->params.scale;
    int input_zero_point = input->params.zero_point;
    quantizeMFCC(mfcc_buffer, quantized_input, N_MFCC * MAX_SEQUENCE_LENGTH, 
                 input_scale, input_zero_point);
    
    // Copy quantized data to input tensor
    memcpy(input->data.int8, quantized_input, N_MFCC * MAX_SEQUENCE_LENGTH * sizeof(int8_t));
    
    // Run inference
    Serial.println("Running inference...");
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        Serial.println("Inference failed!");
        return;
    }
    
    // Get prediction and dequantize
    int8_t prediction_quantized = output->data.int8[0];
    float prediction = (prediction_quantized - output->params.zero_point) * output->params.scale;
    
    // Display results
    Serial.println("=== Prediction Results ===");
    Serial.printf("Predicted water consumption: %.4f liters\r\n", prediction);
    Serial.printf("Input scale: %.6f\r\n", input_scale);
    Serial.printf("Input zero point: %d\r\n", input_zero_point);
    Serial.printf("Output scale: %.6f\r\n", output->params.scale);
    Serial.printf("Output zero point: %d\r\n", output->params.zero_point);
    Serial.println("========================\n");
}

void testWithMockAudio() {
    Serial.println("=== Testing with Mock Audio ===");
    
    // Test with different mock audio patterns
    for (int test = 0; test < 3; test++) {
        Serial.printf("Test %d: ", test + 1);
        
        // Generate different mock audio patterns with temporal variation
        for (int i = 0; i < SAMPLES_PER_CHUNK; i++) {
            float t = (float)i / SAMPLE_RATE;
            
            // Add temporal envelope to simulate water flow patterns
            float envelope = 0.5f + 0.5f * sin(2.0f * PI * 0.5f * t); // Slow modulation
            
            if (test == 0) {
                // Test 1: Low water flow (target ~0.22 liters)
                // Very low amplitude, simple pattern, slow variations
                float base_signal = 0.05f * sin(2.0f * PI * 50.0f * t) +   
                                   0.02f * sin(2.0f * PI * 100.0f * t) +   
                                   0.01f * ((float)random(-200, 200) / 1000.0f);
                audio_buffer[i] = base_signal * envelope * 0.3f; // Reduce overall amplitude
            } else if (test == 1) {
                // Test 2: Medium water flow (target ~1-2 liters)
                // Moderate amplitude and frequency range
                float base_signal = 0.2f * sin(2.0f * PI * 200.0f * t) + 
                                   0.1f * sin(2.0f * PI * 500.0f * t) +
                                   0.05f * sin(2.0f * PI * 1000.0f * t) +
                                   0.03f * ((float)random(-500, 500) / 1000.0f);
                audio_buffer[i] = base_signal * envelope;
            } else {
                // Test 3: High water flow (target ~3+ liters)
                // Higher amplitude, more frequency components, more noise
                float base_signal = 0.4f * sin(2.0f * PI * 300.0f * t) + 
                                   0.3f * sin(2.0f * PI * 800.0f * t) +
                                   0.2f * sin(2.0f * PI * 1500.0f * t) +
                                   0.15f * sin(2.0f * PI * 2500.0f * t) +
                                   0.1f * ((float)random(-800, 800) / 1000.0f);
                audio_buffer[i] = base_signal * envelope * 1.5f; // Increase overall amplitude
            }
        }
        
        // Process and predict
        extractMFCCFeatures(audio_buffer, SAMPLES_PER_CHUNK, mfcc_buffer);
        
        // Print MFCC statistics before normalization
        char label[50];
        sprintf(label, "Test %d MFCC (before norm)", test + 1);
        printMFCCStats(mfcc_buffer, N_MFCC * MAX_SEQUENCE_LENGTH, label);
        
        normalizeMFCC(mfcc_buffer, N_MFCC * MAX_SEQUENCE_LENGTH);
        
        // Print MFCC statistics after normalization
        sprintf(label, "Test %d MFCC (after norm)", test + 1);
        printMFCCStats(mfcc_buffer, N_MFCC * MAX_SEQUENCE_LENGTH, label);
        
        float input_scale = input->params.scale;
        int input_zero_point = input->params.zero_point;
        quantizeMFCC(mfcc_buffer, quantized_input, N_MFCC * MAX_SEQUENCE_LENGTH, 
                     input_scale, input_zero_point);
        
        // Print quantized input statistics
        int8_t min_q = quantized_input[0];
        int8_t max_q = quantized_input[0];
        for (int i = 0; i < N_MFCC * MAX_SEQUENCE_LENGTH; i++) {
            if (quantized_input[i] < min_q) min_q = quantized_input[i];
            if (quantized_input[i] > max_q) max_q = quantized_input[i];
        }
        Serial.printf("Test %d Quantized - Min: %d, Max: %d, Scale: %.6f, Zero: %d\r\n", 
                      test + 1, min_q, max_q, input_scale, input_zero_point);
        
        memcpy(input->data.int8, quantized_input, N_MFCC * MAX_SEQUENCE_LENGTH * sizeof(int8_t));
        
        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status == kTfLiteOk) {
            int8_t prediction_quantized = output->data.int8[0];
            float prediction = (prediction_quantized - output->params.zero_point) * output->params.scale;
            
            // Add description for each test
            if (test == 0) {
                Serial.printf("Low flow simulation - Prediction: %.4f liters\r\n", prediction);
            } else if (test == 1) {
                Serial.printf("Medium flow simulation - Prediction: %.4f liters\r\n", prediction);
            } else {
                Serial.printf("High flow simulation - Prediction: %.4f liters\r\n", prediction);
            }
        } else {
            Serial.println("Inference failed!");
        }
    }
    Serial.println("========================\n");
}

void showHelp() {
    Serial.println("=== Water Prediction Model Commands ===");
    Serial.println("'p' or 'P' - Perform water prediction with mock audio");
    Serial.println("'t' or 'T' - Test with different mock audio patterns");
    Serial.println("'m' or 'M' - Check memory usage");
    Serial.println("'h' or 'H' - Show this help message");
    Serial.println("=====================================\n");
}

void checkMemoryUsage() {
    Serial.println("=== Memory Usage Check ===");
    Serial.printf("Tensor arena size: %d bytes (%.2f KB)\r\n", kTensorArenaSize, kTensorArenaSize / 1024.0);
    Serial.printf("Free heap: %d bytes (%.2f KB)\r\n", ESP.getFreeHeap(), ESP.getFreeHeap() / 1024.0);
    Serial.printf("Free PSRAM: %d bytes (%.2f KB)\r\n", ESP.getFreePsram(), ESP.getFreePsram() / 1024.0);
    Serial.printf("Total heap: %d bytes (%.2f KB)\r\n", ESP.getHeapSize(), ESP.getHeapSize() / 1024.0);
    Serial.printf("Total PSRAM: %d bytes (%.2f KB)\r\n", ESP.getPsramSize(), ESP.getPsramSize() / 1024.0);
    Serial.println("========================\n");
}

void printMFCCStats(float* mfcc_data, int length, const char* label) {
    float min_val = mfcc_data[0];
    float max_val = mfcc_data[0];
    float sum = 0.0f;
    
    for (int i = 0; i < length; i++) {
        if (mfcc_data[i] < min_val) min_val = mfcc_data[i];
        if (mfcc_data[i] > max_val) max_val = mfcc_data[i];
        sum += mfcc_data[i];
    }
    
    float mean = sum / length;
    float range = max_val - min_val;
    
    Serial.printf("%s - Min: %.4f, Max: %.4f, Mean: %.4f, Range: %.4f\r\n", 
                  label, min_val, max_val, mean, range);
}

void setup() {
    Serial.begin(115200);
    while(!Serial);
  
    Serial.println("Water Consumption Prediction Model");
    Serial.println("Initializing TensorFlow Lite Micro Interpreter...");
  
    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    model = tflite::GetModel(model_improved_int8_3_2_1_tflite);
    
    Serial.printf("Model loaded successfully. Size: %d bytes\n", model_improved_int8_3_2_1_tflite_len);
  
    // Check if model and library have compatible schema version,
    // if not, there is a misalignement between TensorFlow version used
    // to train and generate the TFLite model and the current version of library
    Serial.printf("Model version: %d, Library version: %d\n", model->version(), TFLITE_SCHEMA_VERSION);
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
    
    Serial.println("Tensor allocation successful");
    
    // Check actual memory usage
    Serial.printf("Tensor arena size: %d bytes (%.2f KB)\r\n", kTensorArenaSize, kTensorArenaSize / 1024.0);
    Serial.printf("Free heap after allocation: %d bytes\r\n", ESP.getFreeHeap());
  
    // Obtain pointers to the model's input and output tensors.
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    // Print tensor information
    Serial.printf("Input tensor dimensions: ");
    for (int i = 0; i < input->dims->size; i++) {
      Serial.printf("%d", input->dims->data[i]);
      if (i < input->dims->size - 1) Serial.printf("x");
    }
    Serial.printf(" (type: %d)\r\n", input->type);
    
    Serial.printf("Output tensor dimensions: ");
    for (int i = 0; i < output->dims->size; i++) {
      Serial.printf("%d", output->dims->data[i]);
      if (i < output->dims->size - 1) Serial.printf("x");
    }
    Serial.printf(" (type: %d)\r\n", output->type);
  
    Serial.println("Initialization done.");
    Serial.println("");
    Serial.println("Water prediction model ready. Press 'p' to predict or 't' to test with mock audio.");
  
    boot_time = millis();
}

void loop() {
    if (loading) {
        Serial.println("Loading...");
        delay(5000);
        printLoading();
        Serial.println("Loading complete");
        loading = false;
    }

    // Print status every 30 seconds
    /*
    if (millis() - last_status > 30000) {
        checkMemory();
        last_status = millis();
    }
    */

    // Check for serial input
    if (Serial.available()) {
        String inputValue = Serial.readStringUntil('\n');
        inputValue.trim();
        
        if (inputValue == "p" || inputValue == "P") {
            // Perform water prediction with mock audio
            performWaterPrediction();
        } else if (inputValue == "t" || inputValue == "T") {
            // Test with mock audio
            testWithMockAudio();
        } else if (inputValue == "m" || inputValue == "M") {
            // Check memory usage
            checkMemoryUsage();
        } else if (inputValue == "h" || inputValue == "H") {
            // Show help
            showHelp();
        } else {
            Serial.println("Unknown command. Press 'h' for help.");
        }
    }
}