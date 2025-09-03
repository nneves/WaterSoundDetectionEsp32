#include <Arduino.h>

unsigned long boot_time = 0;
unsigned long last_status = 0;
bool loading = true;

void setup() {
  Serial.begin(115200);
  delay(2000);
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
    if (millis() - last_status > 10000) {
        checkMemory();
        last_status = millis();
    }

    delay(1000); // 1 second delay
    Serial.printf("Boot time: %d seconds\r\n", (millis() - boot_time) / 1000);
}
