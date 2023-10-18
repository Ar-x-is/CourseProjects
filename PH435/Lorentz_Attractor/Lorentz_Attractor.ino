#include <MCUFRIEND_kbv.h>

MCUFRIEND_kbv tft;

const int screenWidth = 480; // Width of your TFT screen
const int screenHeight = 320; // Height of your TFT screen

#define BLACK 0x0000
#define WHITE 0xFFFF

const int numPoints = 10000; // Number of points to plot
const float dt = 0.001; // Time step
const float sigma = 20.0; // Lorenz system parameter (scaled down)
const float rho = 50.0; // Lorenz system parameter
const float beta = 8.0 / 3.0; // Lorenz system parameter
float x = 0.1, y = 0.0, z = 0.0; // Initial conditions
bool flag = false;

int px;
int pz;

int prev_px = 0;
int prev_pz = 0;

int last_x[100];
int last_z[100];
int last_index = 0;

void setup() {
  tft.reset();
  uint16_t identifier = tft.readID();
  tft.begin(identifier);
  tft.setRotation(1); // Set screen rotation if needed
  tft.fillScreen(TFT_BLACK);
}

void loop() {
  float dx = sigma * (y - x);
  float dy = x * (rho - z) - y;
  float dz = x * y - beta * z;

  x += dx * dt;
  y += dy * dt;
  z += dz * dt;


  px = map(z, -50, 50, 0, screenWidth);
  pz = map(x, 0, 100, 0, screenHeight);

  // Map z value to a color gradient
  int colorR = map(y, -50, 50, 0, 51000);
  int colorG = map(y, -50, 50, 51000, 0);
  int colorB = map(y, -50, 50, 10000, 30000);
  uint16_t pixelColor = tft.color565(colorR, colorG, colorB);

  tft.drawLine(prev_px, prev_pz, px, pz, pixelColor);

  prev_px = px;
  prev_pz = pz;

  // Erase the oldest line drawn after i == 99
  if (flag) {
    tft.drawLine(last_x[last_index], last_z[last_index], prev_px, prev_pz, BLACK);
  }

  // Store the last 100 points
  last_x[last_index] = px;
  last_z[last_index] = pz;
  last_index = (last_index + 1) % 100;
  if(last_index == 99)
  {
    flag = true;
  }
}
