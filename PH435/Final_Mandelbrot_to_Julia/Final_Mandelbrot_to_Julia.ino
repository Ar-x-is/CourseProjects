#include <Adafruit_GFX.h>
#include <MCUFRIEND_kbv.h>

MCUFRIEND_kbv tft;

#define TFT_WIDTH 320
#define TFT_HEIGHT 320

#include <stdint.h>
#include "TouchScreen.h"

#define YP A2  // must be an analog pin, use "An" notation!
#define XM A3  // must be an analog pin, use "An" notation!
#define YM 8   // can be a digital pin
#define XP 9   // can be a digital pin

// For better pressure precision, we need to know the resistance
// between X+ and X- Use any multimeter to read it
// For the one we're using, its 300 ohms across the X plate
TouchScreen ts = TouchScreen(XP, YP, XM, YM, 300);

bool flag = false;
bool drawflag = false;
float xmin = -1.75;
float xmax = 1.75;
float ymin = -1.75;
float ymax = 1.75;
float x;
float y;
float x_old = x;
float y_old = y;
// Define the properties of the Mandelbrot set
const int maxIterations = 12;
const double zoom = 80;
const double offsetX = 0.0;
const double offsetY = 0.0;
int max_iterations = 63;
int cmap_data[][3] = {{243, 234, 244},
        {237, 231, 239},
        {231, 227, 234},
        {225, 223, 230},
        {219, 220, 226},
        {213, 216, 221},
        {207, 214, 218},
        {202, 209, 215},
        {195, 206, 212},
        {189, 204, 209},
        {183, 201, 208},
        {174, 196, 205},
        {167, 191, 203},
        {160, 187, 200},
        {152, 183, 198},
        {146, 178, 197},
        {140, 173, 196},
        {134, 170, 195},
        {128, 165, 195},
        {122, 165, 194},
        {117, 162, 193},
        {112, 156, 192},
        {108, 151, 191},
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
  Serial.begin(9600);

  tft.reset();
  uint16_t ID = tft.readID();
  tft.begin(ID);
  tft.setRotation(3);
  tft.setTextSize(2);
  tft.fillScreen(TFT_BLACK);
}

void loop() {
  tft.reset();
  uint16_t ID = tft.readID();
  tft.begin(ID);
  tft.setRotation(3);
  tft.setTextSize(2);
  tft.fillScreen(TFT_BLACK);
  // Draw white horizontal line from y=0 to y=320
  tft.drawFastHLine(0, 160, 320, 0xFFFF);
  // Draw white vertical line from x=0 to x=320
  tft.drawFastVLine(160, 0, 320, 0xFFFF);
  generateMandelbrot();
  // a point object holds x y and z coordinates
  while(x == x_old && y == y_old) {
    Point p = ts.getPoint();
    if (p.z > ts.pressureThreshhold && p.z < 1500) {
      x = (670 - p.x)/171.43;
      y = (p.y - 517.5)/218.57;
      Serial.print("X = "); Serial.print(x);
      Serial.print("\tY = "); Serial.print(y);
    } 
  }

  x_old = x;
  y_old = y;
  
  tft.reset();
  tft.begin(ID);
  tft.setRotation(3);
  tft.setTextSize(2);
  tft.fillScreen(TFT_BLACK);
  drawJuliaSet(x,y);
  delay(5000);
}

void generateMandelbrot() {
  for (int y = 0; y < TFT_HEIGHT; y++) {
    for (int x = 0; x < TFT_WIDTH; x++) {
      double zx = 0;
      double zy = 0;
      double cx = (x - TFT_WIDTH / 2.0) / zoom + offsetX;
      double cy = (y - TFT_HEIGHT / 2.0) / zoom + offsetY;

      if (cx == 0 || cy == 0) {
        continue;
      }

      int iteration = 0;
      while (zx * zx + zy * zy < 4 && iteration < maxIterations) {
        double tmp = zx * zx - zy * zy + cx;
        zy = 2 * zx * zy + cy;
        zx = tmp;
        iteration++;
      }

      int color[3] = {cmap_data[iteration]};
      tft.drawPixel(x, y, tft.color565(color[0], color[1], color[2])); 
    }
  }
}
void drawJuliaSet(float x1, float y1) {
  tft.setRotation(0);
  tft.setTextSize(2); // Set text size
  tft.setCursor(60, 2); // Set the text cursor position
  tft.setTextColor(0xFFFF); // Set text color (white)
  tft.print("z = (");
  tft.print(x1, 2); // Print x with 2 decimal places
  tft.print(") + i(");
  tft.print(y1, 2); // Print y with 2 decimal places
  tft.print(")");

  tft.setRotation(1);
  for (int x = 0; x < TFT_WIDTH; x=x+2) {
    for (int y = 0; y < TFT_HEIGHT; y=y+2) {
      float zx = x * (xmax - xmin) / (TFT_WIDTH - 1) + xmin;
      float zy = y * (ymax - ymin) / (TFT_HEIGHT - 1) + ymin;
      float cx = x1;
      float cy = y1;

      int i;
      for (i = 0; i < max_iterations; i++) {
        float x2 = zx * zx;
        float y2 = zy * zy;

        if (x2 + y2 > 4.0) break;

        float xtemp = x2 - y2 + cx;
        zy = 2.0 * zx * zy + cy;
        zx = xtemp;
      }

      int color[3] = {cmap_data[i]};
      tft.drawPixel(x, y, tft.color565(color[0], color[1], color[2]));  
    }
  }
}