# ESP32-S3 Touch AMOLED 1.75-B Project

This PlatformIO project is configured for the Waveshare ESP32-S3 Touch AMOLED 1.75-B development kit.

## Hardware Features

- **Display**: 1.75" Round AMOLED Display (416×416 resolution)
- **Touch**: Capacitive touch screen
- **Microphone**: Dual microphones with ES7210 echo cancellation
- **Audio**: Audio output capability
- **MCU**: ESP32-S3 with 16MB Flash and PSRAM

## Pin Configuration

### Display (SPI)
- MOSI: GPIO 15
- SCLK: GPIO 14  
- CS: GPIO 13
- DC: GPIO 21
- RST: GPIO 47
- MISO: GPIO 38

### Touch
- CS: GPIO 16

### Audio Output (I2S)
- DOUT: GPIO 46
- BCLK: GPIO 9
- LRC: GPIO 45

### Microphone (I2S)
- WS: GPIO 42
- SCK: GPIO 41
- SD: GPIO 2

## Libraries Used

- **TFT_eSPI**: For display control
- **LVGL**: For advanced UI (optional)
- **ESP8266Audio**: For audio playback
- **ESP-DSP**: For digital signal processing

## Getting Started

1. Open this project in PlatformIO
2. Build and upload to your ESP32-S3 Touch AMOLED board
3. The example code will:
   - Initialize the AMOLED display
   - Show touch coordinates when screen is touched
   - Display microphone audio levels
   - Play a test tone every 10 seconds

## Customization

- Modify `src/main.cpp` for your specific application
- Adjust pin configurations in `platformio.ini` if needed
- Update display settings in `include/User_Setup.h`
