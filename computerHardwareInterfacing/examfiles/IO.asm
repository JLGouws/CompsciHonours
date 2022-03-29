init_IO:
  LDI   tmp1, 0xF0
  OUT   DDRD, TMP1                ; set up the d pins data directions
  LDI   tmp1, 0x0C
  OUT   PORTD, TMP1               ; enable pull ups for buttons
  LDI   tmp1, 0xC0                ; 0b11000000
  OUT   GICR, tmp1                ; int0 enabled
  LDI   tmp1, 0x02
  OUT   MCUCR, tmp1               ; falling edge triggered.
  RET

int0_ISR:
  CALL  pause_stepper
  OUT   GICR, ZERO                ; disable external interrupts
  IN    tmp1, TIMSK
  ANDI  tmp1, 0x2F                  ; mask off TIMSK
  ORI   tmp1, 0x40                  ; enable overflow interupts
  OUT   TIMSK, tmp1
  LDI   tmp1, 0x05
  OUT   TCCR2, tmp1               ; start timer2 with prescaler set to /1024
  RETI

int1_ISR:
  CALL  start_stepper
  OUT   GICR, ZERO                ; disable external interrupts
  IN    tmp1, TIMSK
  ANDI  tmp1, 0x2F                  ; mask off TIMSK
  ORI   tmp1, 0x40                  ; enable overflow interupts
  OUT   TIMSK, tmp1
  LDI   tmp1, 0x05
  OUT   TCCR2, tmp1               ; start timer2 with prescaler set to /1024
  RETI

t2_OV_ISR:
  OUT   TCCR2, zero               ; stop counter
  OUT   TCNT2, zero               ; zero counter
  LDI   tmp1, 0xC0
  OUT   GIFR, tmp1                ; clear int0 flag
  OUT   GICR, tmp1                ; int0 re-enabled
  RETI
