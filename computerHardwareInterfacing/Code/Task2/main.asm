;LED-SWITCH - Task2
; Remember to patch PortB0-3 to an LEDs 0-3
; and PORTD0-3 to Switch0-3
.INCLUDEPATH "/usr/share/avra" ;set the inlude path to the correct place
;.DEVICE ATmega16 -- done in m16def.inc
.NOLIST
.INCLUDE "m16def.inc"
.LIST

.def TMP1=R16
.def TMP2=R17
.def ZERO=R0
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
    IN tmp1, pinb
    IN TMP2, PORTD
    ANDI TMP2, 0x0F0
    com TMP1
    ANDI TMP1, 0x0F
    OR TMP1, TMP2
    out portd, tmp1
	RJMP MAIN_LOOP


INITIALISE:
    EOR R0, R0 ; R0 now has the value 0x00
    LDI TMP1, 0X0F ; make the lower 4 bits of portd output
    OUT DDRD, TMP1
    out portd, r0  ; make sure pd0-3 are lo
    ldi TMP1, 0xFF  ;for pull up resistors
    out ddrb, r0 ; make portb inputs
    out portb, TMP1 ;enable pull up resistors
	RET					;Return from subroutine

