; Connect sw0-1 to pd0-1
; Connect PB0-7 to led0-7
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

.org $000		;locate code at address $000
rjmp START		; Jump to the START Label

.org $02A		;locate code past the interupt vectors

START:
	ldi TMP1, LOW(RAMEND)	;initialise the stack pointer
	out SPL, TMP1
	ldi TMP1, HIGH(RAMEND)
	out SPH, TMP1

	RCALL INITIALISE

SAMPLE_DOWN:
	SBIC PIND,0  ; skip if button pushed
	RJMP SAMPLE_DOWN2 ;
	;If we get here the button has been pushed
	ROL COUNT	;increment the counter;
	IN TMP3, SREG
	SBRS TMP3, 0
	RJMP OUT1
	INC COUNT
	CLC
OUT1:
	OUT PORTB, COUNT
SAMPLE_UP:
	SBIS PIND, 0 ; skip if button released
	RJMP SAMPLE_UP
	RJMP SAMPLE_DOWN
SAMPLE_DOWN2:
    SBIC PIND, 1;
    RJMP SAMPLE_DOWN;
    ROR COUNT;
    IN TMP3, SREG
	SBRS TMP3, 0
	RJMP OUT2
	SBR COUNT, 0b10000000
	CLC
OUT2:
    OUT PORTB, COUNT
SAMPLE_UP2:
    SBIS PIND, 1;
    RJMP SAMPLE_UP2;
    RJMP SAMPLE_DOWN2;

INITIALISE:
	; Setup port b as output for LEDs
	; and inital state to all off and set counter to zero
	CLR COUNT
	SER TMP1
    OUT DDRD, COUNT ; make d inputs (count is a handy clear register)
	INC COUNT
	OUT PORTB, COUNT ; 0x00 = all off
	OUT DDRB, TMP1
	SBI PORTD, 0
	SBI PORTD, 1
	RET

