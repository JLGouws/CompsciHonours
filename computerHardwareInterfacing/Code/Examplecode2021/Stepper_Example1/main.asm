;Stepper Example 1
; Hardware connections.
; pd4->drv0
; pd5->drv1
; pd6->drv2
; pd7->drv3
; pd3 -> button0
; pd2 - button1
; We are going to start with a modified example of our debounced external interrupt
; code from the previos example.
; For you to do
;  * Make sure you understand what is going on
;  * Alter the code to back the stepper in the other direction. - maybe making 2 subroutines, turnleft and turnright.
;  * make int0 turn the motor one way and int1 turn the motor the other way.
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

.org $02A		;locate code past the interupt vectors
START:
    ldi TMP1, LOW(RAMEND)	;initialise the stack pointer
	out SPL, TMP1
	ldi TMP1, HIGH(RAMEND)
	out SPH, TMP1
	RCALL INITIALISE_IO ; Call the subroutine INITIALISE
	; we have to set a value in OCR0
    SEI ; enable global interrupts
MAIN_LOOP:
	NOP
	NOP
	RJMP MAIN_LOOP

INITIALISE_IO:
  ldi tmp1, 0xF0;
  out DDRD, tmp1;
  ldi tmp1, 0x10
  mov step, tmp1
  sbi portd, 2 ; pullup on pd2 pin for int 0
  sbi portd, 3 ; pullup on pd3 pin for int 1
  ldi tmp1, 0b11000000
  out GICR, tmp1 ; int0 enabled
  ldi tmp1, 0b00001010
  out mcucr, tmp1 ;falling edge triggered.
	RET					;Return from subroutine

; On overflow of timer we reinit int0, after clearing the flags, stop the timer.
T0ovf_ISR:
 out TCCR0, zero ; stop counter
 out TCNT0, zero ; zero counter
 ldi tmp1, 0b01000000
 out GIFR, tmp1 ; clear int0 flag
                 ; Remove this line and watch the switch bounce "come back" because the flag is set when
                ; the bounce occurs and if we don't clear it, when we re-enable int0 the flag will cause an interrupt
  ldi tmp1, 0b11000000
  out GICR, tmp1 ; int0 re-enabled
  RETI

int0isr:
  call CWISE 
  out gicr, zero ; diable external interrupts
  ldi tmp1, 0x01
  out timsk, tmp1 ; enable t0 overflow
  ldi tmp1, 0b00000101
  out tccr0, tmp1 ; start timer0 with prescaler set to /1024
  RETI

int1isr:
  call CCWISE 
  out gicr, zero ; diable external interrupts
  ldi tmp1, 0x01
  out timsk, tmp1 ; enable t0 overflow
  ldi tmp1, 0b00000101
  out tccr0, tmp1 ; start timer0 with prescaler set to /1024
  RETI

CWISE:
  IN tmp1, PORTD
  andi tmp1, 0x0F ;tmp now contains the masked off values of portD
  lsl step ; increment out variable
  brne outstepc
  ldi tmp2, 0x10
  mov step, tmp2
outstepc:
  or tmp1, step
  out portd, tmp1
  RET

CCWISE:
  IN tmp1, PORTD
  andi tmp1, 0x0F ;tmp now contains the masked off values of portD
  lsr step ; increment out variable
  ldi tmp2, 0xF0
  and tmp2, step
  brne outstepcc
  ldi tmp2, 0x80
  mov step, tmp2
outstepcc:
  or tmp1, step
  out portd, tmp1
  RET
