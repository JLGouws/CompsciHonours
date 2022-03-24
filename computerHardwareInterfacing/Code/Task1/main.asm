;LED-SWITCH - Task1
; This program toggles portB pin 0
; Remember to patch PortB0 to an LED
; and PORTD0 to Switch0
.INCLUDEPATH "/usr/share/avra" ;set the inlude path to the correct place
;.DEVICE ATmega16 -- done in m16def.inc
.NOLIST
.INCLUDE "m16def.inc"
.LIST

.def TMP1=R16		;defines serve the same purpose as in C,
.def TMP2=R17		;before assembly, defined values are substituted
.def TMP3=R18

.cseg			;Tell the assembler that everything below this is in the code segment
.org $000		;locate code at address $000
rjmp START		; Jump to the START Label

.org $02A		;locate code past the interupt vectors
START:
    ldi TMP1, LOW(RAMEND)	;initialise the stack pointer
	out SPL, TMP1
	ldi TMP1, HIGH(RAMEND)
	out SPH, TMP1
	RCALL INITIALISE ; Call the subroutine INITIALISE

MAIN_LOOP:
	; Due to instructions we deviate slightly from
	;	our flow diagram
	SBIC PIND, 0	;skip next instruction if d:0 is low
	CBI  PORTB, 0	;if d:0 is high (nopress) make b:0 low (off)
	SBIS PIND, 0    ;skip if d:0 set (no press)
	SBI	 PORTB, 0	; if pressed take b:0 high (led on)
	NOP

	SBIC PIND, 1	;skip next instruction if d:0 is low
	CBI  PORTB, 1	;if d:0 is high (nopress) make b:0 low (off)
	SBIS PIND, 1    ;skip if d:0 set (no press)
	SBI	 PORTB, 1	; if pressed take b:0 high (led on)
    NOP

    SBIC PIND, 2	;skip next instruction if d:0 is low
	CBI  PORTB, 2	;if d:0 is high (nopress) make b:0 low (off)
	SBIS PIND, 2    ;skip if d:0 set (no press)
	SBI	 PORTB, 2	; if pressed take b:0 high (led on)
	NOP

	CBI  PORTB, 3
	SBIC PIND, 2
	RJMP MAIN_LOOP
    SBIC PIND, 3
	RJMP MAIN_LOOP
	SBI  PORTB, 3

	NOP
	NOP
	RJMP MAIN_LOOP


INITIALISE:
	SBI DDRB, 0			; Make pin0 of portB output
	CBI PORTB, 0		; Set the initial state of pin to low=LED-off
	SBI DDRB, 1         ; Make pin 1 output
	CBI PORTB, 1        ; set initial state of pin off
	SBI DDRB, 2         ; Make pin 1 output
	CBI PORTB, 2        ; set initial state of pin off
	SBI DDRB, 3         ; Make pin 1 output
	CBI PORTB, 3        ; set initial state of pin off
	;Strictly speaking the port value should be set
	;  before making it an output, since this way
	;	the LED will go on for the time it takes to
	;	execute the next instruction.

	CBI DDRD, 0 	;make D:0 an input
	SBI PORTD, 0	; enable pull-up on D:0
	CBI DDRD, 1
	SBI PORTD, 1
    CBI DDRD, 2
	SBI PORTD, 2
    CBI DDRD, 3
	SBI PORTD, 3
	RET					;Return from subroutine

