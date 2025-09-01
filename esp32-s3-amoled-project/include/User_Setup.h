#ifndef USER_SETUP_H
#define USER_SETUP_H

// Waveshare ESP32-S3 Touch AMOLED 1.75-B Display Configuration
#define USER_SETUP_LOADED

// Display driver selection - try ST7789 since RM67162 overrides pins
#define ST7789_DRIVER
// #define RM67162_DRIVER  // This was overriding our pins

// Display resolution - CORRECTED from Waveshare Wiki
#define TFT_WIDTH  466
#define TFT_HEIGHT 466

// Let's try manual SPI setup to avoid pin conflicts
#define USER_SETUP_ID 206

// EXACT pin definitions from Waveshare datasheet pinout table
#define TFT_MISO 3    // DO (MISO) - GPIO3 (from QSPI interface)
#define TFT_MOSI 1    // DI (MOSI) - GPIO1 (from QSPI interface)  
#define TFT_SCLK 2    // SCK (SCLK) - GPIO2 (from QSPI interface)
#define TFT_CS   12   // LCD_CS - GPIO12 (from datasheet LCD column)

// LCD control pins from datasheet
#define TFT_DC   -1   // No DC pin shown in datasheet (QSPI mode might not need it)
#define TFT_RST  39   // LCD_RESET - GPIO39 (from datasheet LCD column)

// Additional LCD pins from datasheet:
// GPIO13: LCD_TE (Tearing Effect - not DC)

// Touch pins for reference (I2C interface)
// GPIO11: TP_INT (Touch interrupt)
// GPIO14: TP_SCL (Touch I2C Clock)  
// GPIO15: TP_SDA (Touch I2C Data)
// GPIO40: TP_RESET

// Wrong pins we were using before:
// #define TFT_MOSI 15, #define TFT_SCLK 14, #define TFT_CS 13, #define TFT_DC 21, #define TFT_RST 47

// Display specific settings
#define TFT_ROTATION 0
#define SPI_FREQUENCY  20000000
#define SPI_READ_FREQUENCY  10000000
#define SPI_TOUCH_FREQUENCY  2500000

// SPI port selection
#define USE_HSPI_PORT

// Additional safety settings
#define TFT_INVERSION_ON
#define TFT_BACKLIGHT_ON 1

// Touch screen support - I2C interface (not SPI)
// Touch uses I2C, not SPI CS pin
// #define TOUCH_CS 16  // Not applicable for I2C touch

// Touch I2C pins from datasheet:
#define TOUCH_SDA 15  // GPIO15: TP_SDA  
#define TOUCH_SCL 14  // GPIO14: TP_SCL
#define TOUCH_INT 11  // GPIO11: TP_INT
#define TOUCH_RST 40  // GPIO40: TP_RESET

// PSRAM support
#define DISABLE_ALL_LIBRARY_WARNINGS

// Color depth
#define TFT_RGB_ORDER TFT_RGB

// FORCE PIN OVERRIDES - Make sure these take precedence
#undef TFT_MISO
#undef TFT_MOSI  
#undef TFT_SCLK
#undef TFT_CS
#undef TFT_DC
#undef TFT_RST

// Re-define with our exact pins
#define TFT_MISO 3    // GPIO3
#define TFT_MOSI 1    // GPIO1
#define TFT_SCLK 2    // GPIO2  
#define TFT_CS   12   // GPIO12
#define TFT_DC   -1   // Not used
#define TFT_RST  39   // GPIO39

#endif // USER_SETUP_H
