;           This file is responsible for handling the stepper motor
; J L Gouws 19G4436
; 
.equ CW   = 0
.equ CCW  = 1

.DSEG
RAM_STEPS: .BYTE  9     ;

.CSEG
init_stepper:
  IN    tmp1, DDRD
  ANDI  tmp1, 0x0F      ; lower bits of the DDRD
  LDI   tmp2, 0xF0
  OR    tmp1, tmp2      ; set D4-D7 to output
  OUT   DDRD, tmp1
  IN    tmp1, PORTD
  ANDI  tmp1, 0x0F      ; lower bits of the PORTD
  LDI   tmp2, 0x10      ; lock motor on first driver
  OR    tmp1, tmp2      ; set D4-D7 to output
  OUT   PORTD, tmp1
  EOR   step, step      ; set step to 0
  RCALL init_timer0
  LDI   ZH, HIGH(2 * STEP_TABLE)
  LDI   ZL, LOW(2 * STEP_TABLE) 
  CALL  read_steps_to_RAM
  RET

init_timer0:
  IN    tmp1, TIMSK
  ANDI  tmp1, 0xFC        ;mask off TIMSK
  ORI   tmp1, 0x02
  OUT   TIMSK, tmp1
  LDI   tmp1, 156         ; output compare 0.125 x 10^{-6} x 256 x 156 = 4.992ms
  OUT   OCR0, tmp1
  LDI   tmp1, 0x00        ; reset timer0s counter
  OUT   TCNT0, tmp1
  RET 

step_motor:
  MOV   slow, ZERO
  MOV   stepCount, ZERO
  LDI   tmp1, 0x04        ; 0b00000100 | clock prescalar 256
  OUT   TCCR0, tmp1       ; start timer
  RET

read_steps_to_RAM:
  LDI   XH, HIGH(RAM_STEPS)
  LDI   XL, LOW(RAM_STEPS)
readPGM:
  LPM   tmp1, Z+
  ST    X+, tmp1
  CPI   tmp1, 0x00
  BRNE  readPGM
  RET

; makes the motor make one step
NEXTSTEP:
  ADD   step, dstep               ; get to the next step
  LDI   tmp1, 0x07
  AND   step, tmp1
  INC   stepCount                 ; get to the next step
  LDI   XH, HIGH(RAM_STEPS)
  LDI   XL, LOW(RAM_STEPS)
  ADD   XL, step
  ADC   XH, zero
  LD    tmp2, X
  IN    tmp1, PORTD
  ANDI  tmp1, 0x0F                  ; tmp1 now contains the masked off values 
                                    ; of portD
  OR    tmp1, tmp2
  OUT   PORTD, tmp1
  RET

t0_OC_ISR:
  LDI   tmp1, 0;
  OUT   TCCR0, tmp1                 ; stop timer 0
  INC   slow
  CP    slow, arg2
  BRNE  contSteps
  MOV   slow, zero
  RCALL nextstep
  CP    stepCount, arg1
  BRNE  contSteps 
  LDI   stepCount, 0x00;
  MOV   slow, zero
  CALL  stepper_done
  RETI                              ; this is done
contSteps:
  LDI   tmp1, 0x00        ; reset timer0s counter
  OUT   TCNT0, tmp1
  LDI   tmp1, 0x04        ; 0b00000100 | clock prescalar 256
  OUT   TCCR0, tmp1       ; revive timer
  RETI

t0_OV_ISR:
  LDI   tmp1, 0;
  OUT   TCCR0, tmp1                 ; stop timer 0
  LDI   tmp1, 0x02        
  OUT   TIFR, tmp1                  ; clear the  output compare flag
  CALL  stepper_done
  RETI

stepper_disable:
  IN    tmp1, TIMSK
  ANDI  tmp1, 0xFC                  ; mask off TIMSK
  ORI   tmp1, 0x01                  ; this stops the timer from resetting on a
                                    ; compare match
  OUT   TIMSK, tmp1

  IN    tmp1, PORTD
  ANDI  tmp1, 0x0F                  ; tmp1 now contains the masked off values 
                                    ; of portD
  OUT   PORTD, tmp1                 ; the lower bits of PORTD are now off
  RET

stepper_enable:
  RCALL init_timer0
  LDI   XH, HIGH(RAM_STEPS)
  LDI   XL, LOW(RAM_STEPS)
  ADD   XL, step
  ADC   XH, zero
  LD    tmp2, X
  LDI   tmp3, 0x00
  CP    step, tmp3
  BRNE  set_motor
  LDI   tmp2, 0x10
set_motor:
  OR    tmp1, tmp2
  OUT   PORTD, tmp1
  RET

;SINGLE_STEPS:     .db 0x09, 0x00, 0x10, 0x10, 0x20, 0x20, 0x40, 0x40, \
;                      0x80, 0x80, 0x00, 0x00
STEP_TABLE:            .db 0x10, 0x30, 0x20, 0x60, 0x40, 0xC0, 0x80, 0x90, 0x00 