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
// TFLM microfrontend for real MFCC-like features
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"


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
constexpr int kTensorArenaSize = 50 * 1024; // 50KB arena in DRAM

// Tensor arena in DRAM (aligned to 16 bytes)
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

unsigned long boot_time = 0;
unsigned long last_status = 0;
bool loading = true;

// Audio processing configuration (matching Python script)
const int CHUNK_DURATION = 3;  // seconds
const int SAMPLE_RATE = 16000;
const int SAMPLES_PER_CHUNK = CHUNK_DURATION * SAMPLE_RATE;
const int N_MFCC = 13;
const int MAX_SEQUENCE_LENGTH = 200;
// Framing to match Python/librosa defaults: 25 ms window, 10 ms hop
const int FRAME_LENGTH_SAMPLES = 400;  // 25ms @ 16kHz
const int HOP_LENGTH_SAMPLES = 160;    // 10ms @ 16kHz

// Small MFCC buffers in DRAM
static float mfcc_buffer[N_MFCC * MAX_SEQUENCE_LENGTH];
static int8_t quantized_input[N_MFCC * MAX_SEQUENCE_LENGTH];

// Microfrontend state and helpers
static FrontendConfig fe_config;
static FrontendState fe_state;
static bool fe_initialized = false;
static int fe_num_mel = 0;

// DCT cache for converting mel bands to MFCCs (DCT-II)
static bool dct_initialized = false;
static int dct_mel_bins = 0;
static float dct_matrix[N_MFCC * 64]; // supports up to 64 mel bins

static void ensureDctMatrix(int mel_bins) {
  if (dct_initialized && dct_mel_bins == mel_bins) return;
  for (int n = 0; n < N_MFCC; n++) {
    for (int k = 0; k < mel_bins; k++) {
      float angle = (float)PI / (float)mel_bins * (k + 0.5f) * n;
      // Orthonormal DCT-II scaling: alpha0 = sqrt(1/M), alpha = sqrt(2/M)
      float alpha = (n == 0) ? sqrtf(1.0f / (float)mel_bins) : sqrtf(2.0f / (float)mel_bins);
      dct_matrix[n * mel_bins + k] = alpha * cosf(angle);
    }
  }
  dct_initialized = true;
  dct_mel_bins = mel_bins;
}

static inline void melToMfcc(const uint16_t* mel, int mel_bins, float* mfcc_out /* length N_MFCC */) {
  ensureDctMatrix(mel_bins);
  for (int n = 0; n < N_MFCC; n++) {
    float acc = 0.0f;
    for (int k = 0; k < mel_bins; k++) {
      acc += ((float)mel[k]) * dct_matrix[n * mel_bins + k];
    }
    mfcc_out[n] = acc;
  }
}

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

// Audio generator mode for tests: 0=water default, 1=low, 2=medium, 3=high, 4=sine1k
static int g_audio_gen_mode = 0;

// Extract MFCC features using TFLM microfrontend (mel + log + PCAN), then DCT-II
void extractMFCCFeatures(float* audio_data, int audio_length, float* mfcc_output) {
  // Clear output buffer
  for (int i = 0; i < N_MFCC * MAX_SEQUENCE_LENGTH; i++) {
    mfcc_output[i] = 0.0f;
  }
  
  // Reset frontend state for a fresh run
  FrontendReset(&fe_state);

  int frames_collected = 0;
  
  // Stream audio in hops to avoid large buffers
  int total_samples = audio_length;
  int generated = 0;
  while (generated < total_samples) {
    int chunk = min(HOP_LENGTH_SAMPLES, total_samples - generated);
    int16_t temp[HOP_LENGTH_SAMPLES];
    for (int i = 0; i < chunk; i++) {
      float v;
      if (audio_data) {
        v = audio_data[generated + i];
      } else {
        float t = (float)(generated + i) / SAMPLE_RATE;
        if (g_audio_gen_mode == 1) {
          float base = 0.05f * sinf(2.0f * PI * 50.0f * t) + 0.02f * sinf(2.0f * PI * 100.0f * t) + 0.01f * ((float)random(-200, 200) / 1000.0f);
          v = base * 0.3f;
        } else if (g_audio_gen_mode == 2) {
          float base = 0.2f * sinf(2.0f * PI * 200.0f * t) + 0.1f * sinf(2.0f * PI * 500.0f * t) + 0.05f * sinf(2.0f * PI * 1000.0f * t) + 0.03f * ((float)random(-500, 500) / 1000.0f);
          v = base;
        } else if (g_audio_gen_mode == 3) {
          float base = 0.4f * sinf(2.0f * PI * 300.0f * t) + 0.3f * sinf(2.0f * PI * 800.0f * t) + 0.2f * sinf(2.0f * PI * 1500.0f * t) + 0.15f * sinf(2.0f * PI * 2500.0f * t) + 0.1f * ((float)random(-800, 800) / 1000.0f);
          v = base * 1.5f;
        } else if (g_audio_gen_mode == 4) {
          v = 0.7f * sinf(2.0f * PI * 1000.0f * t);
        } else {
          v = 0.3f * sinf(2.0f * PI * 100.0f * t) + 0.2f * sinf(2.0f * PI * 500.0f * t) + 0.1f * sinf(2.0f * PI * 2000.0f * t) + 0.05f * ((float)random(-1000, 1000) / 1000.0f);
        }
      }
      if (v > 1.0f) v = 1.0f; if (v < -1.0f) v = -1.0f;
      int32_t s = (int32_t)roundf(v * 32767.0f);
      if (s > 32767) s = 32767; if (s < -32768) s = -32768;
      temp[i] = (int16_t)s;
    }
    size_t num_read = 0;
    FrontendOutput fe_out = FrontendProcessSamples(&fe_state, temp, (size_t)chunk, &num_read);
    generated += (int)num_read;
    if (fe_out.size > 0 && fe_out.values != nullptr) {
      float mfcc_frame[N_MFCC];
      melToMfcc(fe_out.values, (int)fe_out.size, mfcc_frame);
      // Zero the 0th cepstral coefficient (optional parity tweak)
      mfcc_frame[0] = 0.0f;
      if (frames_collected < MAX_SEQUENCE_LENGTH) {
        for (int m = 0; m < N_MFCC; m++) mfcc_output[frames_collected * N_MFCC + m] = mfcc_frame[m];
        frames_collected++;
      } else {
        memmove(mfcc_output, mfcc_output + N_MFCC, (MAX_SEQUENCE_LENGTH - 1) * N_MFCC * sizeof(float));
        for (int m = 0; m < N_MFCC; m++) mfcc_output[(MAX_SEQUENCE_LENGTH - 1) * N_MFCC + m] = mfcc_frame[m];
      }
    }
  }
}

// Normalize MFCC features to match Python pipeline (divide by max abs)
void normalizeMFCC(float* mfcc_data, int length) {
  float max_abs = 0.0f;
  for (int i = 0; i < length; i++) {
    float v = fabsf(mfcc_data[i]);
    if (v > max_abs) max_abs = v;
  }
  if (max_abs <= 1e-6f) {
    return; // avoid division by zero; data already near zero
  }
  const float inv = 1.0f / max_abs;
  for (int i = 0; i < length; i++) {
    mfcc_data[i] *= inv; // scale to roughly [-1, 1]
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
    
    // Generate audio via internal generator and extract MFCCs
    Serial.println("Generating mock audio (internal generator)...");
    Serial.println("Extracting MFCC features...");
    g_audio_gen_mode = 0;
    extractMFCCFeatures(nullptr, SAMPLES_PER_CHUNK, mfcc_buffer);
    
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
        
        // Select generator mode and process
        g_audio_gen_mode = (test == 0 ? 1 : test == 1 ? 2 : 3);
        extractMFCCFeatures(nullptr, SAMPLES_PER_CHUNK, mfcc_buffer);
        
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
    Serial.println("'m' - Check memory usage");
    Serial.println("'s' or 'S' - Generate 1kHz sine and run prediction");
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

void fillBufferWithSine(float freq) {
    // No-op in DRAM-only mode; use generator instead
    (void)freq;
    Serial.println("Using internal sine generator");
}

unsigned long lastSine = 0;
bool continuousSine = false;
void setup() {
    Serial.begin(115200);
    while(!Serial);
  
    Serial.println("Water Consumption Prediction Model");
    Serial.println("Initializing TensorFlow Lite Micro Interpreter...");
  
    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    model = tflite::GetModel(model_improved_int8_3_2_1_tflite);
    
    Serial.printf("Model loaded successfully. Size: %d bytes\r\n", model_improved_int8_3_2_1_tflite_len);
  
    // Check if model and library have compatible schema version,
    // if not, there is a misalignement between TensorFlow version used
    // to train and generate the TFLite model and the current version of library
    Serial.printf("Model version: %d, Library version: %d\r\n", model->version(), TFLITE_SCHEMA_VERSION);
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
      Serial.println("AllocateTensors() failed\r\n");
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
    Serial.printf("Input bytes: %d, scale: %.6f, zero_point: %d\r\n", input->bytes, input->params.scale, input->params.zero_point);
    
    Serial.printf("Output tensor dimensions: ");
    for (int i = 0; i < output->dims->size; i++) {
      Serial.printf("%d", output->dims->data[i]);
      if (i < output->dims->size - 1) Serial.printf("x");
    }
    Serial.printf(" (type: %d)\r\n", output->type);
    Serial.printf("Output bytes: %d, scale: %.6f, zero_point: %d\r\n", output->bytes, output->params.scale, output->params.zero_point);
  
    // Initialize microfrontend config/state (46ms window, 32ms hop, 64 mel channels)
    FrontendFillConfigWithDefaults(&fe_config);
    fe_config.window.size_ms = 46;  // closer to typical librosa default frame length
    fe_config.window.step_size_ms = 32; // hop
    fe_config.filterbank.num_channels = 64; // increase mel resolution
    fe_config.filterbank.lower_band_limit = 20.0f;
    fe_config.filterbank.upper_band_limit = 7600.0f;
    fe_config.pcan_gain_control.enable_pcan = 0; // disable PCAN
    // Disable noise reduction (parity with librosa): set min_signal_remaining = 1.0
    fe_config.noise_reduction.min_signal_remaining = 1.0f;
    fe_config.log_scale.enable_log = 1;          // keep log scale
    fe_config.log_scale.scale_shift = 6;
    FrontendPopulateState(&fe_config, &fe_state, SAMPLE_RATE);
    fe_num_mel = fe_state.filterbank.num_channels;
    Serial.printf("Microfrontend initialized: mel bins=%d, frame=%d, hop=%d\r\n",
                  fe_num_mel, (int)fe_state.window.size, (int)fe_state.window.step);

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
        Serial.println("Press 'h' for help.");
    }
    // Continuous sine playback
    if (continuousSine && millis() - lastSine > 10) {
        fillBufferWithSine(1000.0f);
        lastSine = millis();
    }
    if (Serial.available()) {
        String inputValue = Serial.readStringUntil('\n');
        inputValue.trim();
        if (inputValue == "p" || inputValue == "P") {
            performWaterPrediction();
        } else if (inputValue == "t" || inputValue == "T") {
            testWithMockAudio();
        } else if (inputValue == "m") {
            checkMemoryUsage();
        } else if (inputValue == "s" || inputValue == "S") {
            g_audio_gen_mode = 4;
            extractMFCCFeatures(nullptr, SAMPLES_PER_CHUNK, mfcc_buffer);
            normalizeMFCC(mfcc_buffer, N_MFCC * MAX_SEQUENCE_LENGTH);
            float input_scale = input->params.scale;
            int input_zero_point = input->params.zero_point;
            quantizeMFCC(mfcc_buffer, quantized_input, N_MFCC * MAX_SEQUENCE_LENGTH, input_scale, input_zero_point);
            memcpy(input->data.int8, quantized_input, N_MFCC * MAX_SEQUENCE_LENGTH * sizeof(int8_t));
            TfLiteStatus invoke_status = interpreter->Invoke();
            if (invoke_status == kTfLiteOk) {
                int8_t prediction_quantized = output->data.int8[0];
                float prediction = (prediction_quantized - output->params.zero_point) * output->params.scale;
                Serial.printf("Sine 1kHz - Prediction: %.4f liters\r\n", prediction);
            } else {
                Serial.println("Inference failed!");
            }
        } else if (inputValue == "c") {
            continuousSine = !continuousSine;
            Serial.printf("Continuous sine playback %s\n", continuousSine ? "ENABLED" : "DISABLED");
        } else if (inputValue == "h" || inputValue == "H") {
            showHelp();
        } else {
            Serial.println("Unknown command. Press 'h' for help.");
        }
    }
    // Note: If no sound, try both PA_CTRL HIGH and LOW. Some boards use active LOW for amplifier enable.
}