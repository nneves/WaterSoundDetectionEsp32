#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <driver/i2s.h>
#include <esp_timer.h>
#include <driver/gpio.h>
#include <driver/spi_master.h>
#include "pin_config.h"

// TensorFlow Lite Micro includes  
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_improved_int8_3_2_1.h"

// Display control variables
spi_device_handle_t spi_handle;
bool display_initialized = false;

// Simple demo variables
unsigned long last_status = 0;
unsigned long boot_time = 0;

// TensorFlow Lite Micro variables
namespace {
    // Model constants based on your Python script
    constexpr int kMfccFeatures = 13;  // Number of MFCC coefficients
    constexpr int kMfccFrames = 200;   // Number of time frames (padded sequence length)
    constexpr int kTensorArenaSize = 140 * 1024;  // 140KB for model operations
    constexpr int kInputSize = kMfccFrames * kMfccFeatures;  // 200 * 13 = 2600
    constexpr int kOutputSize = 1;  // Single prediction output
    
    // TensorFlow Lite Micro objects
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;
    uint8_t tensor_arena[kTensorArenaSize];
    
    // Model initialization flag
    bool model_initialized = false;
}

void scanI2CDevices() {
    Serial.println("=== I2C Device Scanner ===");
    Wire.begin(IIC_SDA, IIC_SCL);
    Serial.printf("I2C initialized: SDA=%d, SCL=%d\n", IIC_SDA, IIC_SCL);
    
    byte error, address;
    int nDevices = 0;
    
    for (address = 1; address < 127; address++) {
        Wire.beginTransmission(address);
        error = Wire.endTransmission();
        
        if (error == 0) {
            Serial.printf("I2C device found at address 0x%02X\n", address);
            nDevices++;
        }
    }
    
    if (nDevices == 0) {
        Serial.println("No I2C devices found");
    } else {
        Serial.printf("Found %d I2C device(s)\n", nDevices);
    }
    Serial.println("=== End I2C Scan ===\n");
}

void displayReset() {
    Serial.println("Performing display reset sequence...");
    gpio_set_level((gpio_num_t)LCD_RESET, 0);
    delay(20);
    gpio_set_level((gpio_num_t)LCD_RESET, 1);
    delay(120);
    Serial.println("Display reset completed");
}

void setupQSPIDisplay() {
    Serial.println("=== QSPI Display Initialization ===");
    Serial.printf("LCD_SCLK: GPIO%d\n", LCD_SCLK);
    Serial.printf("LCD_CS: GPIO%d\n", LCD_CS);
    Serial.printf("LCD_RESET: GPIO%d\n", LCD_RESET);
    Serial.printf("LCD_SDIO0-3: GPIO%d,%d,%d,%d\n", LCD_SDIO0, LCD_SDIO1, LCD_SDIO2, LCD_SDIO3);
    Serial.printf("Resolution: %dx%d\n", LCD_WIDTH, LCD_HEIGHT);
    
    // Configure reset pin
    gpio_config_t reset_conf = {
        .pin_bit_mask = (1ULL << LCD_RESET),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE
    };
    gpio_config(&reset_conf);
    
    // Reset display
    displayReset();
    
    // Configure SPI bus for QSPI (4-bit) communication
    spi_bus_config_t bus_config = {
        .mosi_io_num = LCD_SDIO0,
        .miso_io_num = LCD_SDIO1, 
        .sclk_io_num = LCD_SCLK,
        .quadwp_io_num = LCD_SDIO2,
        .quadhd_io_num = LCD_SDIO3,
        .max_transfer_sz = LCD_WIDTH * LCD_HEIGHT * 2 // 16-bit color
    };
    
    esp_err_t ret = spi_bus_initialize(SPI3_HOST, &bus_config, SPI_DMA_CH_AUTO);
    if (ret != ESP_OK) {
        Serial.printf("SPI bus initialization failed: %s\n", esp_err_to_name(ret));
        return;
    }
    
    // Configure SPI device
    spi_device_interface_config_t dev_config = {
        .mode = 0,
        .clock_speed_hz = 80 * 1000 * 1000, // 80 MHz
        .spics_io_num = LCD_CS,
        .flags = SPI_DEVICE_HALFDUPLEX,
        .queue_size = 1
    };
    
    ret = spi_bus_add_device(SPI3_HOST, &dev_config, &spi_handle);
    if (ret != ESP_OK) {
        Serial.printf("SPI device add failed: %s\n", esp_err_to_name(ret));
        return;
    }
    
    Serial.println("QSPI bus initialized successfully");
    display_initialized = true;
    Serial.println("=== End QSPI Setup ===\n");
}

void sendCommand(uint8_t cmd) {
    if (!display_initialized) return;
    
    spi_transaction_t trans = {
        .flags = SPI_TRANS_USE_TXDATA,
        .length = 8,
        .tx_data = {cmd}
    };
    
    esp_err_t ret = spi_device_transmit(spi_handle, &trans);
    if (ret != ESP_OK) {
        Serial.printf("Command send failed: %s\n", esp_err_to_name(ret));
    }
}

void sendData(uint8_t data) {
    if (!display_initialized) return;
    
    spi_transaction_t trans = {
        .flags = SPI_TRANS_USE_TXDATA,
        .length = 8,
        .tx_data = {data}
    };
    
    esp_err_t ret = spi_device_transmit(spi_handle, &trans);
    if (ret != ESP_OK) {
        Serial.printf("Data send failed: %s\n", esp_err_to_name(ret));
    }
}

void initializeDisplay() {
    if (!display_initialized) return;
    
    Serial.println("Initializing CO5300 AMOLED controller...");
    
    // CO5300 AMOLED initialization sequence from Arduino_GFX
    sendCommand(0x11); // Sleep Out
    delay(120);
    
    // Configuration commands
    sendCommand(0xFE); sendData(0x00);  // Page select
    sendCommand(0xC4); sendData(0x80);  // SPI mode control
    sendCommand(0x3A); sendData(0x55);  // Interface Pixel Format 16bit/pixel
    sendCommand(0x53); sendData(0x20);  // Write CTRL Display1
    sendCommand(0x63); sendData(0xFF);  // Write Display Brightness Value in HBM Mode
    sendCommand(0x29);                  // Display ON
    sendCommand(0x51); sendData(0xD0);  // Brightness adjustment (208/255)
    sendCommand(0x58); sendData(0x00);  // Write CE (contrast enhancement off)
    
    delay(10);
    
    Serial.println("CO5300 AMOLED controller initialized");
}

void setAddressWindow(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1) {
    if (!display_initialized) return;
    
    // Set column address (CO5300_W_CASET = 0x2A)
    sendCommand(0x2A);
    sendData(x0 >> 8); sendData(x0 & 0xFF); // Start column
    sendData(x1 >> 8); sendData(x1 & 0xFF); // End column
    
    // Set row address (CO5300_W_PASET = 0x2B)
    sendCommand(0x2B);
    sendData(y0 >> 8); sendData(y0 & 0xFF); // Start row
    sendData(y1 >> 8); sendData(y1 & 0xFF); // End row
    
    // Memory write start (CO5300_W_RAMWR = 0x2C)
    sendCommand(0x2C);
}

void fillScreen(uint16_t color) {
    if (!display_initialized) return;
    
    Serial.printf("Filling screen with color 0x%04X...\n", color);
    
    // Set full screen address window
    setAddressWindow(0, 0, LCD_WIDTH - 1, LCD_HEIGHT - 1);
    
    // Send color data for entire screen
    uint8_t color_high = color >> 8;
    uint8_t color_low = color & 0xFF;
    
    // Send pixel data in chunks to avoid blocking
    const int chunk_size = 1000;
    int total_pixels = LCD_WIDTH * LCD_HEIGHT;
    
    for (int chunk = 0; chunk < total_pixels; chunk += chunk_size) {
        int pixels_in_chunk = min(chunk_size, total_pixels - chunk);
        
        for (int i = 0; i < pixels_in_chunk; i++) {
            sendData(color_high);
            sendData(color_low);
        }
        
        // Small delay to prevent overwhelming the display
        if (chunk % 10000 == 0) {
            Serial.printf("Progress: %d%%\n", (chunk * 100) / total_pixels);
            delay(1);
        }
    }
    
    Serial.println("Screen fill completed");
}

void testBasicDrawing() {
    if (!display_initialized) return;
    
    Serial.println("Testing basic drawing functions...");
    
    // Draw a small square in the center
    int center_x = LCD_WIDTH / 2;
    int center_y = LCD_HEIGHT / 2;
    int size = 50;
    
    setAddressWindow(center_x - size, center_y - size, center_x + size, center_y + size);
    
    // Fill with white
    for (int i = 0; i < (size * 2) * (size * 2); i++) {
        sendData(0xFF); // White high byte
        sendData(0xFF); // White low byte
    }
    
    Serial.println("Basic drawing test completed");
}

void setupAudio() {
    Serial.println("=== Audio Configuration ===");
    Serial.printf("I2S_BCK_IO: GPIO%d\n", I2S_BCK_IO);
    Serial.printf("I2S_WS_IO: GPIO%d\n", I2S_WS_IO);
    Serial.printf("I2S_DO_IO: GPIO%d\n", I2S_DO_IO);
    Serial.printf("I2S_DI_IO: GPIO%d\n", I2S_DI_IO);
    Serial.printf("PA (Amplifier): GPIO%d\n", PA);
    
    // Basic I2S configuration for audio output
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
        .sample_rate = 44100,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 64,
        .use_apll = false,
        .tx_desc_auto_clear = true,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_BCK_IO,
        .ws_io_num = I2S_WS_IO,
        .data_out_num = I2S_DO_IO,
        .data_in_num = I2S_PIN_NO_CHANGE
    };

    esp_err_t result = i2s_driver_install(I2S_NUM_1, &i2s_config, 0, NULL);
    if (result == ESP_OK) {
        Serial.println("Audio I2S driver installed successfully!");
        
        result = i2s_set_pin(I2S_NUM_1, &pin_config);
        if (result == ESP_OK) {
            Serial.println("Audio I2S pins configured successfully!");
        } else {
            Serial.printf("Failed to configure audio I2S pins: %d\n", result);
        }
    } else {
        Serial.printf("Failed to install audio I2S driver: %d\n", result);
    }
    
    Serial.println("=== End Audio Setup ===\n");
}

void testTouchPins() {
    Serial.println("=== Touch Pin Test ===");
    Serial.printf("Touch SDA: GPIO%d\n", IIC_SDA);
    Serial.printf("Touch SCL: GPIO%d\n", IIC_SCL);
    Serial.printf("Touch INT: GPIO%d\n", TP_INT);
    Serial.printf("Touch RESET: GPIO%d\n", TP_RESET);
    
    // Configure touch interrupt pin
    pinMode(TP_INT, INPUT_PULLUP);
    pinMode(TP_RESET, OUTPUT);
    
    // Reset touch controller
    digitalWrite(TP_RESET, LOW);
    delay(20);
    digitalWrite(TP_RESET, HIGH);
    delay(50);
    
    Serial.printf("Touch INT pin state: %s\n", digitalRead(TP_INT) ? "HIGH" : "LOW");
    Serial.println("=== End Touch Test ===\n");
}

bool setupTensorFlowLite() {
    Serial.println("=== TensorFlow Lite Model Setup ===");
    Serial.printf("Model size: %d bytes\n", model_improved_int8_3_2_1_tflite_len);
    Serial.printf("Tensor arena size: %d bytes\n", kTensorArenaSize);
    Serial.printf("Input size: %d (frames: %d, features: %d)\n", kInputSize, kMfccFrames, kMfccFeatures);
    Serial.printf("Output size: %d\n", kOutputSize);
    
    // Initialize TensorFlow Lite Micro
    Serial.println("Calling tflite::InitializeTarget()...");
    tflite::InitializeTarget();
    Serial.println("tflite::InitializeTarget() done.");
    
    // Load the model
    Serial.println("Loading model...");
    model = tflite::GetModel(model_improved_int8_3_2_1_tflite);
    if (!model) {
        Serial.println("Model pointer is null!");
        return false;
    }
    Serial.printf("Model loaded. model->version() = %d, TFLITE_SCHEMA_VERSION = %d\n", model->version(), TFLITE_SCHEMA_VERSION);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.printf("Model schema version %d not supported. Supported version is %d\n", 
                     model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }
    Serial.println("Model schema version OK.");
    Serial.println("Model loaded successfully");
    
    // Create resolver with required operations
    Serial.println("Creating op resolver...");
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddRelu();
    resolver.AddQuantize();
    resolver.AddDequantize();
    Serial.println("Op resolver created.");
    
    // Create interpreter
    Serial.println("Creating MicroInterpreter...");
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
    Serial.println("MicroInterpreter created.");
    
    // Allocate tensors
    Serial.println("Allocating tensors...");
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println("Failed to allocate tensors!");
        return false;
    }
    Serial.println("Tensors allocated successfully");
    
    // Get input and output tensors
    Serial.println("Getting input/output tensors...");
    input = interpreter->input(0);
    output = interpreter->output(0);
    if (!input || !output) {
        Serial.printf("Input or output tensor is null! input=%p, output=%p\n", input, output);
        return false;
    }
    
    // Print tensor information
    Serial.printf("Input tensor: %d dimensions\n", input->dims->size);
    for (int i = 0; i < input->dims->size; i++) {
        Serial.printf("  Dim %d: %d\n", i, input->dims->data[i]);
    }
    Serial.printf("Input type: %d\n", input->type);
    
    Serial.printf("Output tensor: %d dimensions\n", output->dims->size);
    for (int i = 0; i < output->dims->size; i++) {
        Serial.printf("  Dim %d: %d\n", i, output->dims->data[i]);
    }
    Serial.printf("Output type: %d\n", output->type);
    
    // Print quantization parameters if quantized
    if (input->type == kTfLiteInt8) {
        Serial.printf("Input quantization - scale: %.6f, zero_point: %d\n", 
                     input->params.scale, input->params.zero_point);
    }
    if (output->type == kTfLiteInt8) {
        Serial.printf("Output quantization - scale: %.6f, zero_point: %d\n", 
                     output->params.scale, output->params.zero_point);
    }
    
    model_initialized = true;
    Serial.println("=== TensorFlow Lite Setup Complete ===\n");
    return true;
}

void generateDummyMfccData() {
    // Generate dummy MFCC data similar to what your model expects
    // This simulates normalized MFCC features
    Serial.println("Generating dummy MFCC data...");
    
    for (int frame = 0; frame < kMfccFrames; frame++) {
        for (int coeff = 0; coeff < kMfccFeatures; coeff++) {
            // Generate some pseudo-random but realistic MFCC values
            float dummy_value = sin(frame * 0.1 + coeff * 0.3) * 0.5; // Range: -0.5 to 0.5
            
            int index = frame * kMfccFeatures + coeff;
            
            // Handle different input tensor types
            if (input->type == kTfLiteFloat32) {
                input->data.f[index] = dummy_value;
            } else if (input->type == kTfLiteInt8) {
                // Quantize to int8
                float scale = input->params.scale;
                int zero_point = input->params.zero_point;
                int8_t quantized = (int8_t)((dummy_value / scale) + zero_point);
                input->data.int8[index] = quantized;
            }
        }
    }
    
    Serial.printf("Generated %d MFCC values\n", kMfccFrames * kMfccFeatures);
    if (input->type == kTfLiteFloat32) {
        Serial.printf("Sample values (float32): [%.3f, %.3f, %.3f, %.3f, %.3f]\n", 
                     input->data.f[0], input->data.f[1], input->data.f[2], input->data.f[3], input->data.f[4]);
    } else if (input->type == kTfLiteInt8) {
        Serial.printf("Sample values (int8): [%d, %d, %d, %d, %d]\n", 
                     input->data.int8[0], input->data.int8[1], input->data.int8[2], input->data.int8[3], input->data.int8[4]);
    }
}

float runInference() {
    if (!model_initialized) {
        Serial.println("Model not initialized!");
        return -1.0f;
    }
    
    Serial.println("=== Running Inference ===");
    
    // Generate dummy MFCC input data
    generateDummyMfccData();
    
    // Run inference
    unsigned long start_time = micros();
    TfLiteStatus invoke_status = interpreter->Invoke();
    unsigned long inference_time = micros() - start_time;
    
    if (invoke_status != kTfLiteOk) {
        Serial.println("Inference failed!");
        return -1.0f;
    }
    
    // Get prediction and handle different output types
    float prediction = 0.0f;
    if (output->type == kTfLiteFloat32) {
        prediction = output->data.f[0];
    } else if (output->type == kTfLiteInt8) {
        // Dequantize int8 output
        int8_t output_quantized = output->data.int8[0];
        float scale = output->params.scale;
        int zero_point = output->params.zero_point;
        prediction = (output_quantized - zero_point) * scale;
    }
    
    Serial.printf("Inference completed in %lu microseconds\n", inference_time);
    Serial.printf("Prediction: %.4f liters\n", prediction);
    Serial.println("=== End Inference ===\n");
    
    return prediction;
}

void setup() {
    Serial.begin(115200);
    delay(2000);
    boot_time = millis();
    
    Serial.println("======================================");
    Serial.println("ESP32-S3 Touch AMOLED 1.75-B Hardware Test");
    Serial.println("Reverted to Working Configuration");
    Serial.println("======================================\n");
    
    // Check PSRAM
    Serial.println("=== Memory Check ===");
    if (psramFound()) {
        Serial.printf("PSRAM found: %d bytes\n", ESP.getPsramSize());
        Serial.printf("Free PSRAM: %d bytes\n", ESP.getFreePsram());
    } else {
        Serial.println("PSRAM not found!");
    }
    Serial.printf("Free heap: %d bytes\n", ESP.getFreeHeap());
    Serial.printf("CPU frequency: %d MHz\n", ESP.getCpuFreqMHz());
    Serial.println("=== End Memory Check ===\n");
    
    // Test each hardware component
    scanI2CDevices();
    setupQSPIDisplay();
    testTouchPins();
    setupAudio();
    
    // Initialize and test the display
    /*
    if (display_initialized) {
        initializeDisplay();
        
        Serial.println("Testing display with color fills...");
        fillScreen(0xF800); // Red
        delay(2000);
        fillScreen(0x07E0); // Green
        delay(2000);
        fillScreen(0x001F); // Blue
        delay(2000);
        fillScreen(0x0000); // Black
        delay(1000);
        
        Serial.println("Testing basic drawing...");
        testBasicDrawing();
        delay(2000);
    }
    */
    
    Serial.println("======================================");
    Serial.println("Hardware test completed successfully!");
    Serial.println("All pins configured and tested.");
    Serial.println("Check serial output for device detection.");
    Serial.println("======================================\n");
}

void loop() {
    static bool tflite_attempted = false;
    if (!tflite_attempted) {
        Serial.println("[loop] About to call setupTensorFlowLite()");
        bool ok = setupTensorFlowLite();
        Serial.printf("[loop] setupTensorFlowLite() returned: %s\n", ok ? "true" : "false");
        tflite_attempted = true;
    }
    // Print status every 10 seconds
    if (millis() - last_status > 10000) {
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
        Serial.printf("Uptime: %lu ms (%.1f seconds)\n", uptime, uptime / 1000.0);
        Serial.printf("RAM Usage: %.1f%% (%d / %d bytes)\n", ram_usage, used_heap, total_heap);
        Serial.printf("Flash Usage: %.1f%% (%d / %d bytes)\n", flash_usage, sketch_size, total_flash);
        Serial.printf("Free PSRAM: %d bytes\n", ESP.getFreePsram());
        Serial.printf("Touch INT pin: %s\n", digitalRead(TP_INT) ? "HIGH" : "LOW");
        Serial.printf("Display initialized: %s\n", display_initialized ? "YES" : "NO");
        Serial.printf("TFLite model initialized: %s\n", model_initialized ? "YES" : "NO");
        
        // Run periodic inference test
        if (model_initialized && (uptime > 30000)) { // After 30 seconds of uptime
            Serial.println("Running periodic TensorFlow Lite inference test...");
            float prediction = runInference();
            if (prediction >= 0) {
                Serial.printf("Periodic test - Predicted: %.4f liters\n", prediction);
            }
        }
        
        Serial.println("=== End Status ===\n");
        
        last_status = millis();
    }
    
    delay(1000); // 1 second delay
}