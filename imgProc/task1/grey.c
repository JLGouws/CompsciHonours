#include "stdlib.h"
#include "stdio.h"
#include "string.h"

#include "math.h"

#include "common_v4l2.h"

#include <X11/Xlib.h>
#include <X11/Xutil.h>

#define PI 3.14159265358979


char *catchar(char *one, char *two)
{
  char *oneTemp = one;
  while(*(++oneTemp));
  while(*(oneTemp++) = *(two++));
  return one;
}

int getnum(FILE *fp)
{
  char c,
       s = 0;
  while((c = fgetc(fp)) < 32 || s || (c == '#' && s == 1) || (s && c == '\n' && !(s = 0)));
  int num = c - '0';
  while((c = fgetc(fp)) > 32)
    num = 10 * num + c - '0';
  return num;
}

char *strtoint(char *str, int val)
{
  char *tmp = str;
  do
    *(tmp++) = val % 10;
  while (val /= 10);
  *tmp = 0;
  return str;
}

char *copy_image(unsigned char *data, unsigned int width, unsigned int height, unsigned char *dest) {
  unsigned char *tmp = dest;
  for(unsigned register i = 0; i < width * height; i ++)
  {
    *tmp++ = *data++;
    *tmp++ = *data++;
    *tmp++ = *data++;
  }
  return dest;
}

unsigned char *frame_difference(unsigned char *frame1, unsigned char *frame2, unsigned int width, unsigned int height) {
  unsigned char *tmp = frame1;
  for(unsigned register i = 0; i < width * height; i++)
  {
    *tmp++ = *tmp++ = *tmp++ = abs((signed int) *tmp - (signed int) *frame2) > 50 ? *frame2 : 0;
    frame2 += 3;
  }
  return frame1;
}

char *to_grey_red(unsigned char *data, unsigned int width, unsigned int height, unsigned char *grey) {
  unsigned char *tmpData = data - 3,
                *tmpGrey = grey;
  const unsigned lim = width * height;
  for (unsigned register i = 0; i < lim; i++)
    *tmpGrey++ = *tmpGrey++ = *tmpGrey++ = *(data += 3);
  return grey;
}

char *to_grey_green(unsigned char *data, unsigned int width, unsigned int height, unsigned char *grey) {
  unsigned char *tmpData = data - 2,
                *tmpGrey = grey;
  const unsigned lim = width * height;
  for (unsigned register i = 0; i < lim; i++)
    *tmpGrey++ = *tmpGrey++ = *tmpGrey++ = *(data += 3);
  return grey;
}

char *to_grey_blue(unsigned char *data, unsigned int width, unsigned int height, unsigned char *grey) {
  unsigned char *tmpData = data - 1,
                *tmpGrey = grey;
  const unsigned lim = width * height;
  for (unsigned register i = 0; i < lim; i++)
    *tmpGrey++ = *tmpGrey++ = *tmpGrey++ = *(data += 3);
  return grey;
}

char *to_grey(unsigned char *data, unsigned int width, unsigned int height, unsigned char *grey) {
  unsigned char *tmpData = data,
                *tmpGrey = grey;
  const unsigned lim = width * height;
  for (unsigned register i = 0; i < lim; i++)
    *tmpGrey++ = *tmpGrey++ = *tmpGrey++ = (*data++ +  *data++ + *data++) / 3;
  return grey;
}

char *to_grey_correct(unsigned char *data, unsigned int width, unsigned int height, unsigned char *grey) {
  unsigned char *tmpData = data,
                *tmpGrey = grey;
  const unsigned lim = width * height;
  for (unsigned register i = 0; i < lim; i++)
    *tmpGrey++ = *tmpGrey++ = *tmpGrey++ = (30 * (unsigned) *tmpData++ + 59 * (unsigned) *tmpData++ + 11 * (unsigned) *tmpData++ + 50) / 100;
  return grey;
}

unsigned char *rotate(unsigned char *pixels, unsigned int *w1, unsigned int *h1, double th0, unsigned int x0, unsigned int y0)
{
  int w2, h2, x1, y1, yoff;
  double v1, v2;
  w2 = abs(cos(th0) * *w1 + .5) + abs(sin(th0) * *h1 + .5);
  h2 = abs(sin(th0) * *w1 + .5) + abs(cos(th0) * *h1 + .5);
  unsigned char *rotated = (char*) malloc(3 * w2 * h2),
                *t = rotated, *p,
                black[3] = {0, 0, 0},
                white[3] = {255, 255, 255},
                red[3] = {255, 0, 0},
                blue[3] = {0, 0, 255},
                green[3] = {0, 255, 0};

  for (register int y2 = 0; y2 < h2; y2++)
  {
    yoff = (h2 / 2 - y2);
    v1 = sin(th0) * yoff;
    v2 = cos(th0) * (h2 / 2 - y2);
    for (register int x2 = 0; x2 < w2; x2++)
    {
      x1 =  cos(th0) * (x2 - w2 / 2) + v1  + *w1 / 2;
      y1 =  *h1 / 2 + sin(th0) * (x2 - w2 / 2) - v2;
      p = (0 <= x1 && x1 < *w1 && 0 <= y1 && y1 < *h1) ? pixels + 3 * (y1 * *w1 + x1) : black;
      *t++ = *p++;
      *t++ = *p++;
      *t++ = *p++;
    }
  }
  *w1 = w2;
  *h1 = h2;
  return rotated;
}

unsigned char *resize_bilinear(unsigned char *pixels, unsigned int w1, unsigned int h1, unsigned int w2, unsigned int h2)
{
  unsigned char *resized = (char*) malloc(3 * w2 * h2),
                *t = resized, *p, *b;
  unsigned int x_ratio = (int) ((w1 << 16) / w2) + 1,
      y_ratio = (int) ((h1 << 16) / h2) + 1,
      x2, y2, xrat, yrat;
  long ydiff, xdiff;
  for (unsigned register i = 0, yrat = 0; i < h2; i++, yrat += y_ratio)
  {
#define ls16 ((long) 1<<16) 
#define ls32 ((long) 1<<32)
    p = pixels + 3 * (yrat >> 16) * w1;
    b = p + 3 * w1;
    ydiff = (yrat >> 16) + 1 >=  h1 ? 0 : yrat % (ls16);
    for (unsigned register j = 0, xrat = 0; j < w2; j++, xrat += x_ratio)
    {
      x2 = 3 * (xrat >> 16);
      xdiff = x2 + 3 >=  3 * w1 ? 0 : xrat % (ls16);
      *t++ = (b[x2] * (ls16 - xdiff) * (ydiff) + b[x2 + 3] * (xdiff) * (ydiff)
                + p[x2] * (ls16 - xdiff) * (ls16 - ydiff) + p[x2++ + 3] * (xdiff) * (ls16 - ydiff)) / ls32;
      *t++ = (b[x2] * (ls16 - xdiff) * (ydiff) + b[x2 + 3] * (xdiff) * (ydiff)
                + p[x2] * (ls16 - xdiff) * (ls16 - ydiff) + p[x2++ + 3] * (xdiff) * (ls16 - ydiff)) / ls32;
      *t++ = (b[x2] * (ls16 - xdiff) * (ydiff) + b[x2 + 3] * (xdiff) * (ydiff)
                + p[x2] * (ls16 - xdiff) * (ls16 - ydiff) + p[x2++ + 3] * (xdiff) * (ls16 - ydiff)) / ls32;
    }
  }

  return resized;
}

unsigned char *resize(unsigned char *pixels, unsigned int w1, unsigned int h1, unsigned int w2, unsigned int h2)
{
  unsigned char *resized = (char*) malloc(3 * w2 * h2),
                *t = resized, *p;
  unsigned int x_ratio = (int) ((w1 << 16) / w2) + 1,
      y_ratio = (int) ((h1 << 16) / h2) + 1,
      x2, y2, xrat, yrat;
  for (unsigned register i = 0, yrat = 0; i < h2; i++, yrat += y_ratio)
  {
    p = pixels + 3 * (yrat >> 16) * w1;
    for (unsigned register j = 0, xrat = 0; j < w2; j++, xrat += x_ratio)
    {
      x2 = 3 * (xrat >> 16);
      *t++ = p[x2++];
      *t++ = p[x2++];
      *t++ = p[x2++];
    }
  }

  return resized;
}

char *to_bgr_pixmap(char *data, unsigned int width, unsigned int height, char *bgr) {
  const unsigned lim = width * height;
  for (unsigned register i = 0; i < lim; i++)
  {
    bgr[4 * i]      = data[3 * i + 2];
    bgr[4 * i + 1]  = data[3 * i + 1];
    bgr[4 * i + 2]  = data[3 * i];
  }
  return bgr;
}

int *gettriple(FILE *fp, int triple[3])
{
  for (unsigned register i = 0; i < 3; i++)
    i[triple] = fgetc(fp);
  return triple;
}

char *getstr(FILE *fp, char* str)
{
  char *temp = str,
       s = 0;
  while((*temp = fgetc(fp)) < 32 || s || (*temp == '#' && s == 1) || (s && *temp == '\n' && !(s = 0)));
  while((*(++temp) = fgetc(fp)) > 32);
  *temp = 0;
  return str;
}

int main(int argc, char** argv)
{
  char path[100] = "../ppm/",
       *img_name;
  if (argc == 2)
    img_name = argv[1];
  else
  {
    printf("insufficient images\n");
    return 0;
  }
  FILE *fp = fopen(catchar(path, img_name), "rb");
  FILE *out = fopen("output.ppm", "wb");
  fputs("P6\n", out);
  char type[100], s = 1;

  getstr(fp, type);

  int width  = getnum(fp),
      height = getnum(fp),
      max    = getnum(fp),
      triple[3];
  unsigned int
      x_res = 640,
      y_res = 480
  ;
  char *data,
       *rgbImg,
       *fileData,
       *resizedImg,
       *rotatedImg,
       keyPress;
  unsigned char *prev;
  data = (char *) malloc(4 * x_res * y_res);
  rgbImg = (char *) malloc(3 * x_res * y_res);
  fileData = (char *) malloc(3 * width * height);
  prev = (char *) malloc(3 * x_res * y_res);

  fread(fileData, 3 * width * height, 1, fp);
  //to_grey_correct(fileData, width, height, fileData);
//  rotatedImg = rotate(fileData, &width, &height, PI / 4, 0, 0);
  resizedImg = resize_bilinear(fileData, width, height, 3 * width, 3 * height);
  fprintf(out, "%d %d\n%d\n", 3 * width, 3 * height, max);
  fwrite(resizedImg, 27 * width * height, 1, out);
  free(fileData);
  //free(rotatedImg);
  free(resizedImg);
/*
  for(register unsigned i = 0; i < width * height; i++)
  {
    fputc(fileData[3 * i], out);
  }
  //fgets(fileData, 3 * width * height, out);
  for (unsigned register i = 0; i < width * height; i++)
  {
    gettriple(fp, triple);
    int value = (int) (triple[0] + triple[1] + triple[2]) / 3;
//    putc((int) (.3 * triple[0] + .59 * triple[1] + .11 * triple[2] + .5), out);
    putc(value, out);
    putc(value, out);
    putc(value, out);
  }
*/

  Display *d;
  Visual *v;
  Window w;
  GC gc;
  XEvent e;
  XImage *img;
  const char *msg = "Hello, World!";
 
  d = XOpenDisplay(NULL);
  if (d == NULL) {
     fprintf(stderr, "Cannot open display\n");
     exit(1);
  }

  CommonV4l2 common_v4l2;
  char *dev_name = "/dev/video0";
  struct buffer *buffers;

 
  s = DefaultScreen(d);
  v = DefaultVisual(d, s);
  w = XCreateSimpleWindow(d, RootWindow(d, s), 10, 10, x_res, y_res, 1,
                          BlackPixel(d, s), WhitePixel(d, s));

  gc = XCreateGC(d, w, 0, NULL);

  XSelectInput(d, w, ExposureMask | KeyPressMask);
  XMapWindow(d, w);

  CommonV4l2_init(&common_v4l2, dev_name, x_res, y_res);

  CommonV4l2_update_image(&common_v4l2);

  to_grey_correct(CommonV4l2_get_image(&common_v4l2), x_res, y_res, rgbImg);
  to_bgr_pixmap(rgbImg, x_res, y_res, data);
  img = XCreateImage(d, v, 24, ZPixmap, 0, data, x_res, y_res, 32, 0);
 
  while (1) {
    XPutImage(d, w, gc, img, 0, 0, 0, 0, x_res, y_res);
    keyPress = XCheckWindowEvent(d, w, KeyPressMask, &e);
    if (e.type == KeyPress)
      break;
    copy_image(rgbImg, x_res, y_res, prev);
    CommonV4l2_update_image(&common_v4l2);
    to_grey_correct(CommonV4l2_get_image(&common_v4l2), x_res, y_res, rgbImg);
    frame_difference(prev, rgbImg, x_res, y_res);
    to_bgr_pixmap(prev, x_res, y_res, data);
    img = XCreateImage(d, v, 24, ZPixmap, 0, data, x_res, y_res, 32, 0);
  }
 
  XDestroyImage(img);
  XCloseDisplay(d);
  
  free(rgbImg);
  free(prev);
  fclose(fp);
  fclose(out);
  CommonV4l2_deinit(&common_v4l2);
  return 0;
}
