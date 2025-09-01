#include <Arduino.h>
#include <TFT_eSPI.h>
#include <lvgl.h>
#include <driver/i2s.h>
#include <WiFi.h>
#include <SPI.h>

// Display object
TFT_eSPI tft = TFT_eSPI();

// LVGL display buffer - CORRECTED for 466x466 resolution
static lv_disp_draw_buf_t draw_buf;
static lv_color_t buf[466 * 10]; // 10 lines buffer

// LVGL objects for our demo
lv_obj_t *temp_label;
lv_obj_t *humidity_label;
lv_obj_t *battery_bar;
lv_obj_t *wifi_label;
lv_obj_t *touch_label;

// Audio configuration
#define I2S_SAMPLE_RATE     44100
#define I2S_SAMPLE_BITS     16
#define I2S_READ_LEN        1024
#define I2S_CHANNEL_NUM     2

// Microphone I2S configuration
#define MIC_I2S_PORT        I2S_NUM_0
#define MIC_I2S_SAMPLE_RATE 16000
#define MIC_I2S_SAMPLE_BITS 16
#define MIC_I2S_READ_LEN    512

// Demo variables
float demo_temp = 23.5;
float demo_humidity = 65.0;
int demo_battery = 85;
bool wifi_connected = false;
unsigned long last_update = 0;

// LVGL display flush callback
void my_disp_flush(lv_disp_drv_t *disp, const lv_area_t *area, lv_color_t *color_p) {
    uint32_t w = (area->x2 - area->x1 + 1);
    uint32_t h = (area->y2 - area->y1 + 1);

    tft.startWrite();
    tft.setAddrWindow(area->x1, area->y1, w, h);
    tft.pushColors((uint16_t*)&color_p->full, w * h, true);
    tft.endWrite();

    lv_disp_flush_ready(disp);
}

// LVGL touch input callback
void my_touchpad_read(lv_indev_drv_t * indev_driver, lv_indev_data_t * data) {
    uint16_t touchX, touchY;
    bool touched = tft.getTouch(&touchX, &touchY);

    if (touched) {
        data->state = LV_INDEV_STATE_PR;
        data->point.x = touchX;
        data->point.y = touchY;
    } else {
        data->state = LV_INDEV_STATE_REL;
    }
}

void setupDisplay() {
    Serial.println("DEBUG: Starting display setup...");
    delay(500); // Longer delay for power stabilization
    
    // Check if SPI pins are properly configured
    Serial.println("DEBUG: SPI Pin configuration:");
    Serial.printf("  MOSI: %d\n", TFT_MOSI);
    Serial.printf("  SCLK: %d\n", TFT_SCLK);
    Serial.printf("  CS: %d\n", TFT_CS);
    Serial.printf("  DC: %d\n", TFT_DC);
    Serial.printf("  RST: %d\n", TFT_RST);
    
    // Force correct pins before any SPI setup
    Serial.println("DEBUG: Forcing correct pin definitions...");
    
    // Manual SPI setup with EXACT pins from datasheet
    Serial.println("DEBUG: Setting up SPI with datasheet pins...");
    // GPIO1=MOSI, GPIO3=MISO, GPIO2=SCLK, GPIO12=CS
    SPI.begin(2, 3, 1, 12);  // SCLK, MISO, MOSI, CS
    delay(100);
    Serial.println("DEBUG: SPI.begin() with correct pins completed");
    
    // Manual pin setup with exact GPIO numbers
    Serial.println("DEBUG: Setting up control pins...");
    pinMode(12, OUTPUT);  // CS
    pinMode(39, OUTPUT);  // RST
    // DC not used for this display
    
    // Reset display manually using exact GPIO
    Serial.println("DEBUG: Manual display reset...");
    digitalWrite(39, LOW);   // RST = GPIO39
    delay(20);
    digitalWrite(39, HIGH);  // RST = GPIO39
    delay(150);
    Serial.println("DEBUG: Manual reset completed");
    
    // Initialize TFT display with error handling
    Serial.println("DEBUG: Calling tft.init()...");
    try {
        tft.init();
        delay(200);
        Serial.println("DEBUG: tft.init() completed successfully");
    } catch (...) {
        Serial.println("ERROR: tft.init() failed with exception");
        return;
    }
    
    Serial.println("DEBUG: Setting rotation...");
    tft.setRotation(0);
    delay(100);
    Serial.println("DEBUG: Rotation set to 0");
    
    Serial.println("DEBUG: Testing basic drawing...");
    tft.fillScreen(TFT_BLACK);
    delay(100);
    Serial.println("DEBUG: Screen filled black");
    
    // Test basic drawing
    tft.drawPixel(50, 50, TFT_RED);
    delay(50);
    Serial.println("DEBUG: Test pixel drawn");
    
    Serial.println("Display initialized successfully!");
}

void setupLVGL() {
    Serial.println("DEBUG: Starting LVGL initialization...");
    
    Serial.println("DEBUG: Calling lv_init()...");
    lv_init();
    Serial.println("DEBUG: lv_init() completed");
    
    // Initialize display buffer
    Serial.println("DEBUG: Initializing display buffer...");
    lv_disp_draw_buf_init(&draw_buf, buf, NULL, 466 * 10);
    Serial.printf("DEBUG: Display buffer initialized with size: %d bytes\n", sizeof(buf));
    
    // Initialize display driver
    Serial.println("DEBUG: Setting up display driver...");
    static lv_disp_drv_t disp_drv;
    lv_disp_drv_init(&disp_drv);
    disp_drv.hor_res = 466;
    disp_drv.ver_res = 466;
    disp_drv.flush_cb = my_disp_flush;
    disp_drv.draw_buf = &draw_buf;
    Serial.println("DEBUG: Display driver configured, registering...");
    lv_disp_drv_register(&disp_drv);
    Serial.println("DEBUG: Display driver registered");
    
    // Initialize touch input driver
    Serial.println("DEBUG: Setting up touch input driver...");
    static lv_indev_drv_t indev_drv;
    lv_indev_drv_init(&indev_drv);
    indev_drv.type = LV_INDEV_TYPE_POINTER;
    indev_drv.read_cb = my_touchpad_read;
    lv_indev_drv_register(&indev_drv);
    Serial.println("DEBUG: Touch input driver registered");
    
    Serial.println("LVGL initialized successfully!");
}

void createUI() {
    Serial.println("DEBUG: Starting UI creation...");
    
    // Create main screen
    Serial.println("DEBUG: Getting active screen...");
    lv_obj_t *scr = lv_scr_act();
    if (scr == NULL) {
        Serial.println("ERROR: Failed to get active screen!");
        return;
    }
    Serial.println("DEBUG: Active screen obtained, setting background...");
    lv_obj_set_style_bg_color(scr, lv_color_hex(0x000000), 0);
    Serial.println("DEBUG: Background color set to black");
    
    // Title label
    Serial.println("DEBUG: Creating title label...");
    lv_obj_t *title = lv_label_create(scr);
    if (title == NULL) {
        Serial.println("ERROR: Failed to create title label!");
        return;
    }
    Serial.println("DEBUG: Title label created, setting text...");
    lv_label_set_text(title, "ESP32-S3 AMOLED Demo");
    lv_obj_set_style_text_color(title, lv_color_hex(0x00FFFF), 0);
    lv_obj_set_style_text_font(title, &lv_font_montserrat_16, 0);
    lv_obj_align(title, LV_ALIGN_TOP_MID, 0, 10);
    Serial.println("DEBUG: Title label configured and aligned");
    
    // Temperature panel
    lv_obj_t *temp_panel = lv_obj_create(scr);
    lv_obj_set_size(temp_panel, 180, 80);
    lv_obj_align(temp_panel, LV_ALIGN_TOP_LEFT, 10, 50);
    lv_obj_set_style_bg_color(temp_panel, lv_color_hex(0x1E1E1E), 0);
    lv_obj_set_style_border_color(temp_panel, lv_color_hex(0xFF6B35), 0);
    lv_obj_set_style_border_width(temp_panel, 2, 0);
    lv_obj_set_style_radius(temp_panel, 10, 0);
    
    lv_obj_t *temp_title = lv_label_create(temp_panel);
    lv_label_set_text(temp_title, "Temperature");
    lv_obj_set_style_text_color(temp_title, lv_color_hex(0xFF6B35), 0);
    lv_obj_align(temp_title, LV_ALIGN_TOP_MID, 0, 5);
    
    temp_label = lv_label_create(temp_panel);
    lv_label_set_text(temp_label, "23.5°C");
    lv_obj_set_style_text_color(temp_label, lv_color_hex(0xFFFFFF), 0);
    lv_obj_set_style_text_font(temp_label, &lv_font_montserrat_16, 0);
    lv_obj_align(temp_label, LV_ALIGN_CENTER, 0, 10);
    
    // Humidity panel
    lv_obj_t *hum_panel = lv_obj_create(scr);
    lv_obj_set_size(hum_panel, 180, 80);
    lv_obj_align(hum_panel, LV_ALIGN_TOP_RIGHT, -10, 50);
    lv_obj_set_style_bg_color(hum_panel, lv_color_hex(0x1E1E1E), 0);
    lv_obj_set_style_border_color(hum_panel, lv_color_hex(0x4ECDC4), 0);
    lv_obj_set_style_border_width(hum_panel, 2, 0);
    lv_obj_set_style_radius(hum_panel, 10, 0);
    
    lv_obj_t *hum_title = lv_label_create(hum_panel);
    lv_label_set_text(hum_title, "Humidity");
    lv_obj_set_style_text_color(hum_title, lv_color_hex(0x4ECDC4), 0);
    lv_obj_align(hum_title, LV_ALIGN_TOP_MID, 0, 5);
    
    humidity_label = lv_label_create(hum_panel);
    lv_label_set_text(humidity_label, "65.0%");
    lv_obj_set_style_text_color(humidity_label, lv_color_hex(0xFFFFFF), 0);
    lv_obj_set_style_text_font(humidity_label, &lv_font_montserrat_16, 0);
    lv_obj_align(humidity_label, LV_ALIGN_CENTER, 0, 10);
    
    // Battery bar
    lv_obj_t *bat_label = lv_label_create(scr);
    lv_label_set_text(bat_label, "Battery Level");
    lv_obj_set_style_text_color(bat_label, lv_color_hex(0x95E1D3), 0);
    lv_obj_align(bat_label, LV_ALIGN_LEFT_MID, 10, -40);
    
    battery_bar = lv_bar_create(scr);
    lv_obj_set_size(battery_bar, 380, 20);
    lv_obj_align(battery_bar, LV_ALIGN_LEFT_MID, 10, -10);
    lv_bar_set_range(battery_bar, 0, 100);
    lv_bar_set_value(battery_bar, demo_battery, LV_ANIM_OFF);
    lv_obj_set_style_bg_color(battery_bar, lv_color_hex(0x2D2D2D), LV_PART_MAIN);
    lv_obj_set_style_bg_color(battery_bar, lv_color_hex(0x4CAF50), LV_PART_INDICATOR);
    lv_obj_set_style_radius(battery_bar, 10, 0);
    
    // WiFi status
    wifi_label = lv_label_create(scr);
    lv_label_set_text(wifi_label, "WiFi: Disconnected");
    lv_obj_set_style_text_color(wifi_label, lv_color_hex(0xFFC107), 0);
    lv_obj_align(wifi_label, LV_ALIGN_LEFT_MID, 10, 30);
    
    // Touch coordinates
    touch_label = lv_label_create(scr);
    lv_label_set_text(touch_label, "Touch: No touch detected");
    lv_obj_set_style_text_color(touch_label, lv_color_hex(0xE91E63), 0);
    lv_obj_align(touch_label, LV_ALIGN_LEFT_MID, 10, 60);
    
    // Create some interactive buttons
    lv_obj_t *btn1 = lv_btn_create(scr);
    lv_obj_set_size(btn1, 120, 50);
    lv_obj_align(btn1, LV_ALIGN_BOTTOM_LEFT, 10, -10);
    lv_obj_set_style_bg_color(btn1, lv_color_hex(0x9C27B0), 0);
    lv_obj_set_style_radius(btn1, 25, 0);
    
    lv_obj_t *btn1_label = lv_label_create(btn1);
    lv_label_set_text(btn1_label, "Button 1");
    lv_obj_center(btn1_label);
    
    lv_obj_t *btn2 = lv_btn_create(scr);
    lv_obj_set_size(btn2, 120, 50);
    lv_obj_align(btn2, LV_ALIGN_BOTTOM_RIGHT, -10, -10);
    lv_obj_set_style_bg_color(btn2, lv_color_hex(0xFF5722), 0);
    lv_obj_set_style_radius(btn2, 25, 0);
    
    lv_obj_t *btn2_label = lv_label_create(btn2);
    lv_label_set_text(btn2_label, "Button 2");
    lv_obj_center(btn2_label);
    
    // Status indicator (circular)
    lv_obj_t *status_arc = lv_arc_create(scr);
    lv_obj_set_size(status_arc, 80, 80);
    lv_obj_align(status_arc, LV_ALIGN_BOTTOM_MID, 0, -80);
    lv_arc_set_range(status_arc, 0, 100);
    lv_arc_set_value(status_arc, 75);
    lv_obj_set_style_arc_color(status_arc, lv_color_hex(0x2196F3), LV_PART_INDICATOR);
    lv_obj_set_style_arc_width(status_arc, 8, LV_PART_INDICATOR);
    lv_obj_set_style_arc_color(status_arc, lv_color_hex(0x424242), LV_PART_MAIN);
    lv_obj_set_style_arc_width(status_arc, 8, LV_PART_MAIN);
    
    lv_obj_t *arc_label = lv_label_create(status_arc);
    lv_label_set_text(arc_label, "75%");
    lv_obj_set_style_text_color(arc_label, lv_color_hex(0x2196F3), 0);
    lv_obj_center(arc_label);
    
    Serial.println("DEBUG: UI creation completed successfully!");
    Serial.println("LVGL UI created successfully!");
}

void updateDemoData() {
    // Update demo values with some animation
    static float temp_delta = 0.1;
    static float hum_delta = 0.2;
    static int bat_delta = 1;
    
    demo_temp += temp_delta;
    if (demo_temp > 30.0 || demo_temp < 20.0) temp_delta = -temp_delta;
    
    demo_humidity += hum_delta;
    if (demo_humidity > 80.0 || demo_humidity < 40.0) hum_delta = -hum_delta;
    
    demo_battery += bat_delta;
    if (demo_battery > 100 || demo_battery < 20) bat_delta = -bat_delta;
    
    // Update UI elements
    char temp_str[20];
    sprintf(temp_str, "%.1f°C", demo_temp);
    lv_label_set_text(temp_label, temp_str);
    
    char hum_str[20];
    sprintf(hum_str, "%.1f%%", demo_humidity);
    lv_label_set_text(humidity_label, hum_str);
    
    lv_bar_set_value(battery_bar, demo_battery, LV_ANIM_ON);
    
    // Update WiFi status (toggle for demo)
    wifi_connected = !wifi_connected;
    if (wifi_connected) {
        lv_label_set_text(wifi_label, "WiFi: Connected");
        lv_obj_set_style_text_color(wifi_label, lv_color_hex(0x4CAF50), 0);
    } else {
        lv_label_set_text(wifi_label, "WiFi: Disconnected");
        lv_obj_set_style_text_color(wifi_label, lv_color_hex(0xFFC107), 0);
    }
}

void updateTouchDisplay() {
    uint16_t x, y;
    static uint16_t last_x = 0, last_y = 0;
    static bool was_touched = false;
    
    boolean touched = tft.getTouch(&x, &y);
    
    if (touched) {
        if (!was_touched || abs(x - last_x) > 5 || abs(y - last_y) > 5) {
            char touch_str[50];
            sprintf(touch_str, "Touch: X=%d, Y=%d", x, y);
            lv_label_set_text(touch_label, touch_str);
            last_x = x;
            last_y = y;
        }
        was_touched = true;
    } else {
        if (was_touched) {
            lv_label_set_text(touch_label, "Touch: No touch detected");
            was_touched = false;
        }
    }
}

void setupAudioOutput() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
        .sample_rate = I2S_SAMPLE_RATE,
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
        .bck_io_num = 9,
        .ws_io_num = 45,
        .data_out_num = 46,
        .data_in_num = I2S_PIN_NO_CHANGE
    };

    esp_err_t result = i2s_driver_install(I2S_NUM_1, &i2s_config, 0, NULL);
    if (result == ESP_OK) {
        Serial.println("Audio output I2S driver installed successfully!");
    } else {
        Serial.printf("Failed to install audio I2S driver: %d\n", result);
    }

    result = i2s_set_pin(I2S_NUM_1, &pin_config);
    if (result == ESP_OK) {
        Serial.println("Audio output I2S pins configured successfully!");
    } else {
        Serial.printf("Failed to configure audio I2S pins: %d\n", result);
    }
}

void setupMicrophone() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = MIC_I2S_SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 64,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = 41,
        .ws_io_num = 42,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = 2
    };

    esp_err_t result = i2s_driver_install(MIC_I2S_PORT, &i2s_config, 0, NULL);
    if (result == ESP_OK) {
        Serial.println("Microphone I2S driver installed successfully!");
    } else {
        Serial.printf("Failed to install microphone I2S driver: %d\n", result);
    }

    result = i2s_set_pin(MIC_I2S_PORT, &pin_config);
    if (result == ESP_OK) {
        Serial.println("Microphone I2S pins configured successfully!");
    } else {
        Serial.printf("Failed to configure microphone I2S pins: %d\n", result);
    }
}

void setup() {
    Serial.begin(115200);
    delay(2000);
    
    Serial.println("=== ESP32-S3 Touch AMOLED 1.75-B LVGL Demo Starting ===");
    
    // Check PSRAM
    Serial.println("DEBUG: Checking PSRAM...");
    if (psramFound()) {
        Serial.printf("DEBUG: PSRAM found: %d bytes\n", ESP.getPsramSize());
    } else {
        Serial.println("WARNING: PSRAM not found!");
    }
    
    // Check free memory
    Serial.printf("DEBUG: Free heap: %d bytes\n", ESP.getFreeHeap());
    
    // LCD INITIALIZATION DISABLED FOR DEBUGGING
    Serial.println("=== STEP 1: LCD INITIALIZATION DISABLED ===");
    Serial.println("LCD initialization skipped to avoid crashes");
    Serial.println("Using serial debug mode only");
    
    /*
    // Initialize display (DISABLED - causing crashes)
    Serial.println("=== STEP 1: Initializing display ===");
    setupDisplay();
    Serial.println("=== STEP 1: Display setup completed ===");
    
    // Skip LVGL for now to test basic display
    Serial.println("=== STEP 2: LVGL DISABLED FOR TESTING ===");
    Serial.println("Testing basic display functionality only...");
    
    // Test basic display drawing
    Serial.println("Testing basic display operations...");
    tft.fillScreen(TFT_BLUE);
    delay(1000);
    tft.fillScreen(TFT_RED);
    delay(1000);
    tft.fillScreen(TFT_GREEN);
    delay(1000);
    tft.fillScreen(TFT_BLACK);
    
    tft.setTextColor(TFT_WHITE, TFT_BLACK);
    tft.setTextSize(2);
    tft.drawString("ESP32-S3", 50, 100);
    tft.drawString("AMOLED Test", 50, 130);
    tft.drawString("Display OK!", 50, 160);
    Serial.println("Basic display test completed!");
    
    // Initialize LVGL (disabled for testing)
    Serial.println("=== STEP 2: Initializing LVGL ===");
    setupLVGL();
    Serial.println("=== STEP 2: LVGL setup completed ===");
    
    // Create UI
    Serial.println("=== STEP 3: Creating UI ===");
    createUI();
    Serial.println("=== STEP 3: UI creation completed ===");
    */
    
    // Initialize audio (optional)
    Serial.println("=== STEP 4: Initializing audio ===");
    setupAudioOutput();
    setupMicrophone();
    Serial.println("=== STEP 4: Audio setup completed ===");
    
    Serial.println("=== ALL SETUP COMPLETE ===");
    Serial.println("Touch the screen to interact with the demo!");
    Serial.printf("DEBUG: Final free heap: %d bytes\n", ESP.getFreeHeap());
}

void loop() {
    // LCD DISABLED - Serial debug mode only
    
    // Print periodic status every 5 seconds
    static unsigned long last_status = 0;
    if (millis() - last_status > 5000) {
        Serial.printf("=== SYSTEM STATUS ===\r\n");
        Serial.printf("Uptime: %lu ms\r\n", millis());
        Serial.printf("Free heap: %d bytes\r\n", ESP.getFreeHeap());
        Serial.printf("PSRAM free: %d bytes\r\n", ESP.getFreePsram());
        Serial.printf("CPU frequency: %d MHz\r\n", ESP.getCpuFreqMHz());
        Serial.printf("Flash size: %d bytes\r\n", ESP.getFlashChipSize());
        Serial.printf("=== END STATUS ===\r\n\r\n");
        last_status = millis();
    }
    
    /*
    // LVGL and touch code (disabled)
    // Handle LVGL tasks
    lv_timer_handler();
    
    // Test touch functionality directly
    uint16_t x, y;
    boolean touched = tft.getTouch(&x, &y);
    
    if (touched) {
        Serial.printf("Touch detected: X=%d, Y=%d\n", x, y);
        // Draw a small circle where touched
        tft.fillCircle(x, y, 5, TFT_YELLOW);
    }
    */
    
    delay(100); // Small delay
    Serial.printf("Hello World\r\n");
}