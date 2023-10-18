#include <Adafruit_GFX.h>
#include <MCUFRIEND_kbv.h>

MCUFRIEND_kbv tft;

#define TFT_WIDTH 320
#define TFT_HEIGHT 320

float xmin = -1.75;
float xmax = 1.75;
float ymin = -1.75;
float ymax = 1.75;
int max_iterations = 63;
int cmap_data[][3] = {{182, 134, 209},
{174, 115, 194},
{165, 95, 179},
{157, 76, 164},
{149, 56, 149},
{140, 37, 134},
{132, 17, 119},
{143, 24, 124},
{154, 31, 129},
{166, 38, 134},
{177, 45, 139},
{189, 52, 144},
{200, 59, 149},
{212, 66, 154},
{223, 73, 159},
{235, 80, 164},
{246, 87, 169},
{255, 95, 174},
{255, 127, 191},
{255, 159, 209},
{255, 191, 226},
{255, 223, 244},
{255, 255, 255},
{105, 145, 190},
{102, 140, 188},
{100, 136, 187},
{98, 132, 185},
{96, 127, 184},
{95, 121, 182},
{94, 115, 180},
{93, 109, 178},
{92, 105, 176},
{91, 100, 172},
{90, 93, 166},
{87, 84, 142},
{83, 79, 117},
{78, 81, 81},
{66, 61, 67},
{55, 54, 51},
{44, 41, 40},
{36, 33, 34},
{29, 27, 27},
{23, 18, 21},
{19, 14, 17},
{19, 14, 17},
{50, 11, 51},
{61, 15, 58},
{73, 18, 65},
{93, 23, 73},
{110, 28, 78},
{156, 34, 79},
{156, 58, 79},
{156, 70, 79},
{175, 83, 81},
{184, 98, 85},
{189, 109, 91},
{206, 130, 97},
{214, 147, 105},
{220, 166, 114},
{229, 181, 122},
{243, 204, 130},
{255, 216, 133},
{255, 226, 134}};

void setup() {
  // Serial.begin(9600);
  tft.reset();
  uint16_t ID = tft.readID();
  tft.begin(ID);
  tft.setRotation(3);
  tft.setTextSize(2);
  tft.fillScreen(TFT_BLACK);
}

void loop() {
  float a = 0.0;  // Set the initial 'a' value
  float a_step = 0.314;  // Adjust the step size as needed
  
  for (int frame = 0; frame < 20; frame++) { // Adjust the number of frames as needed
    drawJuliaSet(a);
    a += a_step;
  }
}

void drawJuliaSet(float a) {
  for (int x = 0; x < TFT_WIDTH; x++) {
    for (int y = 0; y < TFT_HEIGHT; y++) {
      float zx = x * (xmax - xmin) / (TFT_WIDTH - 1) + xmin;
      float zy = y * (ymax - ymin) / (TFT_HEIGHT - 1) + ymin;

      float cx = 0.7885 * cos(a);
      float cy = 0.7885 * sin(a);

      int i;
      for (i = 0; i < max_iterations; i++) {
        float x2 = zx * zx;
        float y2 = zy * zy;

        if (x2 + y2 > 4.0) break;

        float xtemp = x2 - y2 + cx;
        zy = 2.0 * zx * zy + cy;
        zx = xtemp;
      }

      // int color[3] = {cmap_data[i]};
      // Serial.print(color[0]);
      // Serial.print(" ");
      // Serial.println(cmap_data[i][0]);
      tft.drawPixel(x, y, tft.color565(cmap_data[i][0], cmap_data[i][1], cmap_data[i][2]));  
    }
  }
}