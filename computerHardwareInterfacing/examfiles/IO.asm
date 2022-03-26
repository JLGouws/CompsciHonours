init_IO:
  LDI   TMP1, 0xF0
  OUT   DDRD, TMP1   ; set up the d pins data directions
  LDI   TMP1, 0x0C
  OUT   PORTD, TMP1  ; lock the step motor and enable pull ups for buttons
