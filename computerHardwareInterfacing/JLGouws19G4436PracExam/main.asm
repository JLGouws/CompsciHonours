.INCLUDEPATH "/usr/share/avra" ;set the inlude path to the correct place
;.DEVICE ATmega16
.NOLIST
.INCLUDE "m16def.inc"
.LIST

.def MUL_LOW=R0
.def MUL_HIGH=R1
.def ZERO=R2
.def step=R3
.def dstep=R4
.def slow=R5
.def retReg=R6
.def tmp1=R16       ; my code is a bit brittle somewhere
.def tmp2=R17       ; it seems like tmp1 and tmp2 have to
.def tmp3=R18       ; r16 and r17
.def arg1=R19
.def arg2=R20
.def tasknum=R21
.def stepCount=R22

.CSEG                             ; start the code segment
.org 0x000                        ; locate code at address $000
  RJMP  START                     ; Jump to the START Label
.org INT0addr
  JMP   INT0_ISR
.org INT1addr
  JMP   INT1_ISR
.org URXCaddr
  JMP   urxc_isr
.org UTXCaddr
  JMP   utxc_isr
.org OC0addr
  JMP   t0_OC_ISR
.org OVF2addr
  JMP   t2_OV_ISR
.org ADCCaddr
  JMP   ADC_ISR
.org OC1Aaddr
  JMP   t1_OCA_ISR
.org OVF1addr
  JMP   t1_OV_ISR

.org $02A		                      ; locate code past the interupt vectors

START:
  LDI   tmp1, LOW(RAMEND)	
  OUT   SPL, tmp1
  LDI   tmp1, HIGH(RAMEND)        ; initialise the stack pointer
  OUT   SPH, tmp1

  EOR   zero, zero                ; make zero register, well zero

  ; set task num to zero
  LDI   tasknum, 0x00
  LDI   XH, HIGH(TASK_NUM_RAM)
  LDI   XL, LOW(TASK_NUM_RAM)
  ST    X, tasknum                ; make the task number buffer zero

  ; initialize the microcontroller
  CALL  init_LCD
  CALL  init_EEP
  CALL  init_UART
  CALL  init_stepper
  CALL  init_IO
  CALL  init_watchdog
  SEI
  CALL  do_task
MAIN_LOOP:
  NOP
  NOP
  WDR
  RJMP MAIN_LOOP

; files to include
.include "IO.asm"
.include "LCD.asm"
.include "UART.asm"
.include "EEP.asm"
.include "stepperMotor.asm"
.include "taskHandler.asm"
.include "ADC.asm"
.include "watchdog.asm"
