#include <Adafruit_GFX.h>
#include <MCUFRIEND_kbv.h>

MCUFRIEND_kbv tft;

// Define colors
#define BLACK 0x0000
#define RED   0xF800
#define GREEN 0x07E0
#define BLUE  0x001F
#define WHITE 0xFFFF

// Define the vertices of the triangle
int16_t vertexY[3] = {0, 0, tft.width()};
int16_t vertexX[3] = {0, tft.height(), tft.height()};

// Initialize the current point as a random point inside the triangle
int16_t currentX = random(160, 241);
int16_t currentY = random(20, 221);

void setup() {
  tft.begin(0x9481); // Initialize the LCD
  tft.setRotation(0); // Set the display rotation (0 for portrait)
  tft.fillScreen(BLACK); // Fill the screen with black

  // Draw the vertices of the triangle
  tft.fillCircle(vertexX[0], vertexY[0], 5, RED);
  tft.fillCircle(vertexX[1], vertexY[1], 5, GREEN);
  tft.fillCircle(vertexX[2], vertexY[2], 5, BLUE);
}

void loop() {
  // Choose a random vertex
  int16_t randomVertex = random(3);

  // Calculate the midpoint between the current point and the chosen vertex
  currentX = (currentX + vertexX[randomVertex]) / 2;
  currentY = (currentY + vertexY[randomVertex]) / 2;

  // Draw the new point
  tft.drawPixel(currentX, currentY, WHITE);

  // You can adjust the delay to control the animation speed
  delay(10); // Delay between points (adjust as needed)
}

