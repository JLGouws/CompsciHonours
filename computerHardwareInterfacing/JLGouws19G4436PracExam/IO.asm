;                             IO.asm
; J L Gouws
; File for handling the input and output of microcontroler

.DSEG
lightValue: .BYTE 1

.CSEG
init_IO:
  IN    tmp1, DDRD
  ANDI  tmp1, 0xF0
  OUT   DDRD, tmp1                ; set up the d pins data directions
  LDI   tmp1, 0x7E
  OUT   DDRA, tmp1                ; set up the d pins data directions
  IN    tmp1, PORTD
  ANDI  tmp1, 0xF0
  ORI   tmp1, 0x0C
  OUT   PORTD, tmp1               ; enable pull ups for buttons
  LDI   tmp1, 0x0A
  OUT   MCUCR, tmp1               ; both falling edge triggered.
  IN    tmp1, TIMSK               ; timer 1 for PWM
  ANDI  tmp1, 0xC3
  ORI   tmp1, 0x14                ; enable overflow and OCA interupts for timer 1
  OUT   TIMSK, tmp1 
  RET

enable_buttons:
  LDI   tmp1, 0xC0                ; 0b11000000
  OUT   GICR, tmp1                ; int0 and int1 enabled
  OUT   GIFR, tmp1                ; clear int0 flag and in1 flags
  RET

disable_buttons:
  OUT   GICR, zero                ; int0 and int1 disabled
  RET

output_lights:
  OUT   TCCR1A, zero
  LDI   tmp1, 0x01                ; prescalar of 1, 8 makes the timer too slow
  OUT   TCCR1B, tmp1
  LDI   tmp2, 0x3F                ; multiply the lights to get a nice range
                                  ; of brightness
  MUL   tmp2, arg1 
  ADD   MUL_LOW, tmp2
  ADC   MUL_HIGH, zero
  OUT   OCR1AH, MUL_HIGH          ; set the ouput compare value
  OUT   OCR1AL, MUL_LOW
  LDI   XH, HIGH(lightValue)      ; save this value to RAM
  LDI   XL, LOW(lightValue)
  LSR   arg1
  ANDI  arg1, 0x7E
  ST    X, arg1
  RET

clear_lights:
  OUT   TCCR1A, zero              ; stop timer 1 for pulse width modulation
  OUT   TCCR1B, zero
  IN    tmp1, PORTA               ;turn off lights
  ANDI  tmp1, 0x81
  OUT   PORTA, tmp1
  RET

int0_ISR:
  CALL  pause_stepper
  OUT   GICR, ZERO                ; disable external interrupts
  IN    tmp1, TIMSK
  ANDI  tmp1, 0x3F                ; mask off TIMSK
  ORI   tmp1, 0x40                ; enable overflow interupts
  OUT   TIMSK, tmp1
  LDI   tmp1, 0x05
  OUT   TCCR2, tmp1               ; start timer2 with prescaler set to /1024
  RETI

int1_ISR:
  CALL  start_stepper
  OUT   GICR, ZERO                ; disable external interrupts
  IN    tmp1, TIMSK
  ANDI  tmp1, 0x3F                ; mask off TIMSK
  ORI   tmp1, 0x40                ; enable overflow interupts
  OUT   TIMSK, tmp1
  LDI   tmp1, 0x05
  OUT   TCCR2, tmp1               ; start timer2 with prescaler set to /1024
  RETI

t2_OV_ISR:
  OUT   TCCR2, zero               ; stop counter
  OUT   TCNT2, zero               ; zero counter
  RCALL enable_buttons
  RETI

t1_OCA_ISR:
  IN    tmp1, PORTA               ;clear lights on output compare
  ANDI  tmp1, 0x81
  OUT   PORTA, tmp1
  RETI

t1_OV_ISR:
  LDI   XH, HIGH(lightValue)      ; get adc readout again
  LDI   XL, LOW(lightValue)
  LD    tmp1, X
  IN    tmp2, PORTA               ; get PORTA values
  ANDI  tmp2, 0x81                ; mask off values
  OR    tmp1, tmp2
  OUT   PORTA, tmp1               ; set the lights
  RETI
