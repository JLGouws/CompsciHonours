;Counter Example 3 - debouncing a switch
; For you to do * Make sure you understand what is going on. 

.INCLUDEPATH "/usr/share/avra" ;set the inlude path to the correct place

.NOLIST
.INCLUDE "m16def.inc"
.LIST

.def zero=R0        ; reserve R0 as our ZERO register
.def count=R1
.def TMP1=R16		;defines serve the same purpose as in C,
.def TMP2=R17		;before assembly, defined values are substituted
.def TMP3=R18

.cseg			;Tell the assembler that everything below this is in the code segment
.org $000		;locate code at address $000
jmp START		; Jump to the START Label
.org INT0addr
jmp int0isr
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
ldi tmp1, 0x0F;
out DDRA, tmp1;
clr count
sbi portd, 2 ; pullup on pd2 pin for int 0
ldi tmp1, 0b01000000
out GICR, tmp1 ; int0 enabled
ldi tmp1, 0b00000010
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
ldi tmp1, 0b01000000
out GICR, tmp1 ; int0 re-enabled
    RETI

int0isr:
inc count ; increment out variable
out porta, count
;uncomment the code section below to enable our switch debouncing
out gicr, zero ; diable external interrupts
ldi tmp1, 0x01
out timsk, tmp1 ; enable t0 overflow
ldi tmp1, 0b00000101
out tccr0, tmp1 ; start timer0 with prescaler set to /1024
;---end of debouncing
   RETI
