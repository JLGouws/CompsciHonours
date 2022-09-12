#include "time.h"
#include "stdlib.h"
#include "stdio.h"
#include "string.h"

#include "math.h"


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

unsigned char *hideRed(unsigned char *pixels, char *message, unsigned int w, unsigned int h)
{
  unsigned char *tmp = pixels - 3;
  do
    *(tmp += 3) = *message++;
  while  (*(message - 1) != 0 && tmp - pixels < 3 * w * h);
  return pixels;
}

char *findRed(unsigned char *pixels, char *message, unsigned int w, unsigned int h)
{
  unsigned char *tmp2 = pixels - 3;
  char *tmp1 = message;
  do
    *tmp1++ = *(tmp2 += 3);
  while((tmp2 - pixels < 3 * w * h) && *(tmp1 - 1) != 0);
  return message;
}

char *findStraight(unsigned char *pixels, char *message, unsigned int w, unsigned int h)
{
  unsigned char *tmp2 = pixels;
  char *tmp1 = message;
  do
    *tmp1++ = *tmp2++;
  while((tmp2 - pixels < 3 * w * h) && *(tmp1 - 1) != '\0');
  return message;
}

unsigned char *hideStraight(char *pixels, char *message, unsigned int w, unsigned int h)
{
  char *tmp = pixels;
  do
    *tmp++ = *message++;
  while  (*(message - 1) != 0 && tmp - pixels < 3 * w * h);
  return pixels;
}

unsigned char *findBits(unsigned char *pixels, char *message, unsigned int w, unsigned int h)
{
  unsigned char *tmp =  message,
                *tmp2 = pixels;
  do
  {
    *tmp = 0;
    for(unsigned register i = 0; i < 8; i++)
    {
      *tmp <<= 1;
      *tmp |= *pixels++ % 2;
    }
  }
  while  (*tmp++ != '\0' && pixels - tmp2 + 8 < 3 * w * h);
  return message;
}

unsigned char *hideBits(unsigned char *pixels, unsigned char *message, unsigned int w, unsigned int h)
{
  unsigned char *tmp = pixels, cs;
  message--;
  do
  {
    cs = *++message;
    for(unsigned register i = 0; i < 8; i++, cs <<= 1)
    {
      *tmp &= 0xFE;
      *tmp++ |= cs / 128;
    }
  }
  while  (*message != '\0' && tmp - pixels + 8 < 3 * w * h);
  return pixels;
}

unsigned char *encrypt(unsigned char *pixels, unsigned char *o11, unsigned char *o21, unsigned int w, unsigned int h)
{
  srand(time(NULL));   // Initialization, should only be called once.
  unsigned char px[16] =  {0x09, 0x0C, 0x05, 0x06, 0x03},
                cp = *pixels++,
                p1,
                p2,
                *o12 = o11 + (2 * w + 7) / 8,
                *o22 = o21 + (2 * w + 7) / 8;
  int count = 0, count2 = 6, dest, index, bit, r;
  for(unsigned register i = 0; i < h; i++)
  {
    if(count != 0)
    {
      cp = *pixels++;
      count = 0;
    }
    for(unsigned register j = 0; j < w; j++)
    {
      r = ((long) 5 * rand()) / RAND_MAX;
      p1 = px[r];
      p2 = cp / 128 == 0 ? p1 : ~p1;

      *o11 |= (p1 >> 2) << count2;
      *o21 |= (p2 >> 2) << count2;


      *o12 |= (p1 % 4) << count2;
      *o22 |= (p2 % 4) << count2;
      cp <<= 1;
      count = (count + 1) % 8;
      if(count == 0) {
        cp = *pixels++;
      }
      if (count2 == 0) {
        count2 = 8;
        o11++;
        o21++;
        o12++;
        o22++;
      }
      count2 -= 2;
    }
      count2 = 6;
      o12++;
      o22++;
    o11 = o12;
    o21 = o22;
    o12 += (2 * w + 7) / 8;
    o22 += (2 * w + 7) / 8;
  }
}

unsigned char *decrypt(unsigned char *pixels, unsigned char *o1, unsigned char *o2, unsigned int w, unsigned int h)
{
  for(unsigned register i = 0; i < (w + 7) / 8 * h; i++)
  {
    *pixels++ = *o1++ | *o2++;
  }
}

unsigned char *frame_difference(unsigned char *frame1, unsigned char *frame2, unsigned int width, unsigned int height) {
  unsigned char *tmp = frame1;
  for(unsigned register i = 0; i < width * height; i++)
  {
    *tmp++ = *tmp++ = *tmp++ = abs((signed int) *tmp - (signed int) *frame2) > 25 ? abs((signed int) *tmp - (signed int) *frame2) : 0;
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
       *img_name,
       *message;
  if (argc == 3)
  {
    img_name = argv[1];
    message  = argv[2];
  }
  else
  {
    printf("insufficient input\n");
    return 0;
  }
  FILE *fp = fopen(catchar(path, img_name), "rb");
  FILE *out3 = fopen("out.ppm", "wb");
  fputs("P6\n", out3);
  char type[100], s = 1, msgBuff[1000];

  getstr(fp, type);

  int width  = getnum(fp),
      height = getnum(fp),
      max = getnum(fp);
  unsigned char *data,
                *decrypted;
  unsigned char prev;
  data     = (char *) malloc(3 * width * height);

  fread(data, 3 * width * height, 1, fp);
  fprintf(out3, "%d %d\n%d\n", width, height,max);
  hideBits(data, message, width, height);
  printf("\n");
  findBits(data, msgBuff, width, height);
  printf("%s\n", msgBuff);
  fwrite(data, 3 * width * height, 1, out3);
  free(data);
  fclose(fp);
  fclose(out3);
  return 0;
}
