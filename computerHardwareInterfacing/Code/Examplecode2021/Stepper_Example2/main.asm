;Stepper Example 2
; Hardware connections.
; pd4->drv0
; pd5->drv1
; pd6->drv2
; pd7->drv3
; pd3 -> button0
; pd2 - button1
; We are going to start with a modified example of our previous stepper motor code
; For you to do
;  * Make sure you understand what is going on
;  * Alter the code to have int0 start or stop the motor.
;  * Alter the code so that int1 will add 0x10 to the oc0 value to step through speeds
;  *  (and wrap around to a small value again)
.INCLUDEPATH "/usr/share/avra" ;set the inlude path to the correct place

.NOLIST
.INCLUDE "m16def.inc"
.LIST

.def zero=R0        ; reserve R0 as our ZERO register
.def step=R1
.def TMP1=R16		;defines serve the same purpose as in C,
.def TMP2=R17		;before assembly, defined values are substituted
.def TMP3=R18

.cseg			;Tell the assembler that everything below this is in the code segment
.org $000		;locate code at address $000
jmp START		; Jump to the START Label
.org INT0addr
jmp int0isr
.org INT1addr
jmp int1isr
.org OVF0addr   ; locate next instruction at Timer0's overflow interrupt vector
jmp T0ovf_ISR      ; Jump to the interrupt service routine
.org OC0addr
jmp T0OC_ISR

.org $02A		;locate code past the interupt vectors
START:
	ldi TMP1, HIGH(RAMEND)
	out SPH, TMP1
  ldi TMP1, LOW(RAMEND)	;initialise the stack pointer
	out SPL, TMP1
	CALL INITIALISE_IO ; Call the subroutine INITIALISE
	call initialise_t0 ; call subroutine to init t0 with output comparevalue
    SEI ; enable global interrupts
MAIN_LOOP:
	NOP
	NOP
	RJMP MAIN_LOOP

Initialise_t0:
    ldi tmp1, 0x60 ; try a range of values here and observe the motor speed. As an exercise determine the # of steps per
                    ; second that the motor is performing for each value
    out ocr0, tmp1 ; setup oc0
    ldi tmp1, 0b00000010
    out timsk, tmp1 ; enable t0 ocint - we use the output compare intterupt in this case
                    ; because we will set the timer to CTC mode
    ldi tmp1, 0b00001101; ctc mode, no OC pin use, /1024 prescaler
    out TCCR0, tmp1
    ret

INITIALISE_IO:
    ldi tmp1, 0xF0;
    out DDRD, tmp1;
    ldi tmp1, 0x10
    mov step, tmp1
    sbi portd, 2 ; pullup on pd2 pin for int 0
    sbi portd, 3 ; pullup on pd2 pin for int 1
    ldi tmp1, 0b11000000
    out GICR, tmp1 ; int0 enabled
    ldi tmp1, 0b00001010
    out mcucr, tmp1 ;falling edge triggered.
    RET					;Return from subroutine

; On overflow of timer we reinit int0, after clearing the flags, stop the timer.
T0ovf_ISR:
;call Nextstep
    RETI
T0OC_ISR:
  call Nextstep
  RETI

int0isr:
  ldi tmp1, 0b00001101
  IN TMP2, TCCR0
  EOR tmp1, TMP2
  OUT TCCR0, TMP1
  RETI

NEXTSTEP:
  IN tmp1, PORTD
  andi tmp1, 0x0F ;tmp now contains the masked off values of portD
  lsl step ; increment out variable
  brne outstep
  ldi tmp2, 0x10
  mov step, tmp2
outstep:
  or tmp1, step
  out portd, tmp1
  RET

int1isr:
  ldi tmp1, 0x00
  ldi tmp2, 0x10
  out TCCR0, TMP1 ;stop timer
  in tmp1, OCR0
  CPI TMP1, 0xF0
  BRNE FIN
  LDI TMP1, 0x50
FIN:
  ADD TMP1, TMP2
  OUT OCR0, TMP1
  ldi tmp1, 0x0d;start timer again
  out TCCR0, TMP1 ;stop timer
  RETI

