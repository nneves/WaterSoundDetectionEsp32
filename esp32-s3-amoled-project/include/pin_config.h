#pragma once

// Pin definitions for ESP32-S3 Touch AMOLED 1.75-B
// Based on Arduino demo configuration

// LCD QSPI interface pins
#define LCD_SDIO0 4
#define LCD_SDIO1 5
#define LCD_SDIO2 6
#define LCD_SDIO3 7
#define LCD_SCLK 38
#define LCD_CS 12
#define LCD_RESET 39
#define LCD_WIDTH 466
#define LCD_HEIGHT 466

// Touch I2C interface pins
#define IIC_SDA 15
#define IIC_SCL 14
#define TP_INT 11
#define TP_RESET 40

// Audio ES8311 pins
#define I2S_MCK_IO 16
#define I2S_BCK_IO 9
#define I2S_DI_IO 10
#define I2S_WS_IO 45
#define I2S_DO_IO 8

// Audio amplifier pin
#define PA 46

// SD card pins
#define SDMMC_CLK 2
#define SDMMC_CMD 1
#define SDMMC_DATA 3
#define SDMMC_CS 41

// Power management
#define XPOWERS_CHIP_AXP2101
