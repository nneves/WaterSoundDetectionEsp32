#ifndef LV_CONF_H
#define LV_CONF_H

#include <stdint.h>

/* Graphical settings */
#define LV_COLOR_DEPTH     16
#define LV_COLOR_16_SWAP   0
#define LV_COLOR_SCREEN_TRANSP    0
#define LV_COLOR_MIX_ROUND_OFS    (LV_COLOR_DEPTH == 32 ? 0: 128)
#define LV_COLOR_CHROMA_KEY    lv_color_hex(0x00ff00)         /*Images pixels with this color will not be drawn if they are chroma keyed)*/

/* Memory settings */
#define LV_MEM_CUSTOM      1
#if LV_MEM_CUSTOM == 0
#define LV_MEM_SIZE        (48U * 1024U)          /*[bytes]*/
#define LV_MEM_ATTR        
#define LV_MEM_ADR          0     /*0: unused*/
#else       /*LV_MEM_CUSTOM*/
#define LV_MEM_CUSTOM_INCLUDE  <stdlib.h>   /*Header for the dynamic memory function*/
#define LV_MEM_CUSTOM_ALLOC   malloc
#define LV_MEM_CUSTOM_FREE    free
#define LV_MEM_CUSTOM_REALLOC realloc
#endif     /*LV_MEM_CUSTOM*/

/* Input device settings */
#define LV_INDEV_DEF_READ_PERIOD          30                     /*[ms]*/
#define LV_INDEV_DEF_DRAG_LIMIT           10                     /*[px]*/
#define LV_INDEV_DEF_DRAG_THROW           10                     /*[%]*/
#define LV_INDEV_DEF_LONG_PRESS_TIME      400                    /*[ms]*/
#define LV_INDEV_DEF_LONG_PRESS_REP_TIME  100                    /*[ms]*/
#define LV_INDEV_DEF_GESTURE_LIMIT        50                     /*[px]*/
#define LV_INDEV_DEF_GESTURE_MIN_VELOCITY 3

/* Use a custom tick source */
#define LV_TICK_CUSTOM     1
#if LV_TICK_CUSTOM == 1
#define LV_TICK_CUSTOM_INCLUDE  "Arduino.h"         /*Header for the system time function*/
#define LV_TICK_CUSTOM_SYS_TIME_EXPR (millis())     /*Expression evaluating to current system time in ms*/
#endif   /*LV_TICK_CUSTOM*/

/* Feature usage */
#define LV_USE_ANIMATION        1
#define LV_USE_SHADOW           1
#define LV_USE_BLEND_MODES      1
#define LV_USE_OPA_SCALE        1
#define LV_USE_IMG_TRANSFORM    1
#define LV_USE_GROUP            1
#define LV_USE_GPU              0
#define LV_USE_FILESYSTEM       0
#define LV_USE_USER_DATA        1

/* Image decoder and cache */
#define LV_IMG_CACHE_DEF_SIZE   0
#define LV_USE_LARGE_COORD      0

/* Built-in image formats */
#define LV_USE_IMG_BMP          1
#define LV_USE_IMG_PNG          1
#define LV_USE_IMG_SJPG         1

/* Typography */
#define LV_USE_FONT_COMPRESSED  1
#define LV_USE_FONT_SUBPX       0

/* Font sizes to enable */
#define LV_FONT_MONTSERRAT_8     0
#define LV_FONT_MONTSERRAT_10    0
#define LV_FONT_MONTSERRAT_12    1
#define LV_FONT_MONTSERRAT_14    1
#define LV_FONT_MONTSERRAT_16    1
#define LV_FONT_MONTSERRAT_18    1
#define LV_FONT_MONTSERRAT_20    1
#define LV_FONT_MONTSERRAT_22    1
#define LV_FONT_MONTSERRAT_24    1
#define LV_FONT_MONTSERRAT_26    0
#define LV_FONT_MONTSERRAT_28    0
#define LV_FONT_MONTSERRAT_30    0
#define LV_FONT_MONTSERRAT_32    0
#define LV_FONT_MONTSERRAT_34    0
#define LV_FONT_MONTSERRAT_36    0
#define LV_FONT_MONTSERRAT_38    0
#define LV_FONT_MONTSERRAT_40    0
#define LV_FONT_MONTSERRAT_42    0
#define LV_FONT_MONTSERRAT_44    0
#define LV_FONT_MONTSERRAT_46    0
#define LV_FONT_MONTSERRAT_48    0

/* Widget usage */
#define LV_USE_ARC       1
#define LV_USE_ANIMIMG   1
#define LV_USE_BAR       1
#define LV_USE_BTN       1
#define LV_USE_BTNMATRIX 1
#define LV_USE_CANVAS    1
#define LV_USE_CHECKBOX  1
#define LV_USE_DROPDOWN  1
#define LV_USE_IMG       1
#define LV_USE_LABEL     1
#define LV_USE_LINE      1
#define LV_USE_LIST      1
#define LV_USE_METER     1
#define LV_USE_MSGBOX    1
#define LV_USE_ROLLER    1
#define LV_USE_SLIDER    1
#define LV_USE_SPAN      1
#define LV_USE_SPINBOX   1
#define LV_USE_SPINNER   1
#define LV_USE_SWITCH    1
#define LV_USE_TEXTAREA  1
#define LV_USE_TABLE     1
#define LV_USE_TABVIEW   1
#define LV_USE_TILEVIEW  1
#define LV_USE_WIN       1

/* Themes */
#define LV_USE_THEME_DEFAULT    1
#define LV_USE_THEME_BASIC      1
#define LV_USE_THEME_MONO       1

/* Layouts */
#define LV_USE_LAYOUT_FLEX 1
#define LV_USE_LAYOUT_GRID 1

/* Examples */
#define LV_BUILD_EXAMPLES 0

/* Debug */
#define LV_USE_LOG      1
#if LV_USE_LOG
  #define LV_LOG_LEVEL    LV_LOG_LEVEL_WARN
  #define LV_LOG_PRINTF   0
  #define LV_LOG_USE_TIMESTAMP 1
#endif  /*LV_USE_LOG*/

/* Asserts */
#define LV_USE_ASSERT_NULL          1
#define LV_USE_ASSERT_MALLOC        1
#define LV_USE_ASSERT_STYLE         0
#define LV_USE_ASSERT_MEM_INTEGRITY 0
#define LV_USE_ASSERT_OBJ           0

#define LV_ASSERT_HANDLER_INCLUDE   <stdint.h>
#define LV_ASSERT_HANDLER   while(1);   /*Halt by default*/

#endif /*LV_CONF_H*/
