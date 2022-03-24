.INCLUDEPATH "/usr/share/avra" ;set the inlude path to the correct place
;.DEVICE ATmega16 -- done in m16def.inc
.NOLIST
.INCLUDE "m16def.inc"
.LIST

.def TMP1=R16		;
.def TMP2=R17		;
.def TMP3=R18
.def COUNT=R19	;store the count in this register

.cseg			;Tell the assembler that everything below this is in the code segment

.org 0x000		;locate code at address $000
rjmp START		; Jump to the START Label
.org INT0addr
rjmp INT0_ROUTINE
.org INT1addr
rjmp INT1_ROUTINE
.org INT2addr
rjmp INT2_ROUTINE

.org $02A		;locate code past the interupt vectors

START:
	ldi TMP1, LOW(RAMEND)	;initialise the stack pointer
	out SPL, TMP1
	ldi TMP1, HIGH(RAMEND)
	out SPH, TMP1

	RCALL INITIALISE
	SEI
MAIN_LOOP:
	NOP
	NOP
	NOP
	RJMP MAIN_LOOP

INITIALISE:
	; Setup port b as output for LEDs
	; and inital state to all off and set counter to zero
	CLR TMP1;
	OUT DDRD, TMP1	; ensure that portd is an input
	SBI PORTD, 2		; enable pull-ups on int0
	SBI PORTD, 3		; enable pull-ups on int1
	CBI DDRB, 2
	SBI PORTB, 2
	OUT PORTA, TMP1	; porta all low
    SER TMP1
	OUT DDRA, TMP1	; porta output
	CLR COUNT		; clear the counter
	LDI TMP1, 0x09	; int0 falling, int1 rising
	OUT	MCUCR, TMP1
	LDI TMP1, 0b11100000
	OUT GICR, TMP1
	RET

INT0_ROUTINE:
;int 0 is to increment the counter
	PUSH TMP1		;save tmp1
	IN TMP1, SREG
	PUSH TMP1		; save sreg
	DEC COUNT		; decrement the counter
	OUT PORTA, count	; output
	POP TMP1
	OUT SREG, TMP1	;resotre sreg
	POP TMP1		;resotr tmp1
	RETI

INT1_ROUTINE:
;int 1 is to increment the counter
	PUSH TMP1
	IN TMP1, SREG
	PUSH TMP1
	INC COUNT
	OUT PORTA, count
	POP TMP1
	OUT SREG, TMP1
	POP TMP1
	RETI

INT2_ROUTINE:
    PUSH TMP1
    IN TMP1, SREG
    PUSH TMP1
    IN TMP1, GICR
    LDI TMP2, 0b01000000
    EOR TMP1, TMP2
    OUT GICR, TMP1
    IN TMP1, GIFR
    SBR TMP1, 6
    OUT GIFR, TMP1
    POP TMP1
    OUT SREG, TMP1
    POP TMP1
    RETI
