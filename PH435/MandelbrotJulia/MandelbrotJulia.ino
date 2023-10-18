#include <MCUFRIEND_kbv.h>
#include <TouchScreen.h>

#define YP A2  // must be an analog pin, use "An" notation!
#define XM A3  // must be an analog pin, use "An" notation!
#define YM 8   // can be a digital pin
#define XP 9   // can be a digital pin
#define	BLACK  0x0000
#define TFT_WIDTH 320
#define TFT_HEIGHT 320

// Create an instance of the TFT display
MCUFRIEND_kbv tft;
// For better pressure precision, we need to know the resistance
// between X+ and X- Use any multimeter to read it
// For the one we're using, its 300 ohms across the X plate
TouchScreen ts = TouchScreen(XP, YP, XM, YM, 300);

// Define the properties of the Mandelbrot set
const int maxIterations = 50;
const double zoom = 100;
const double offsetX = 0.0;
const double offsetY = 0.0;

void setup() {
  // Initialize the TFT display
  tft.begin(0x9481);
  tft.setRotation(3); // Adjust the rotation as needed
  tft.fillScreen(BLACK);

  // Generate and display the Mandelbrot set
  generateMandelbrot();
}

void loop() {
  // Not used in this example
  Point p = ts.getPoint();

  // select the c value based on p.x, p.y
  // generateJuliaSet(c);

  // delay(10000);

  // go back to mandelbrot

}

void generateMandelbrot() {
  for (int y = 0; y < TFT_HEIGHT; y++) {
    for (int x = 0; x < TFT_WIDTH; x++) {
      double zx = 0;
      double zy = 0;
      double cx = (x - TFT_WIDTH / 2.0) / zoom + offsetX;
      double cy = (y - TFT_HEIGHT / 2.0) / zoom + offsetY;

      int iteration = 0;
      while (zx * zx + zy * zy < 4 && iteration < maxIterations) {
        double tmp = zx * zx - zy * zy + cx;
        zy = 2 * zx * zy + cy;
        zx = tmp;
        iteration++;
      }

      uint16_t color = tft.color565(iteration % 2 * 334545, iteration % 3 * 16436, iteration % 5 * 83453);
      tft.drawPixel(x, y, color);
    }
  }
}