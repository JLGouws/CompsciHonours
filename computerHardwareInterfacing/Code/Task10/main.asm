; LCD
;Portc0-7 connected to LCD_data0-7
;PB0 - E
;PB1 - rw
;PB2 - rs

.INCLUDEPATH "/usr/share/avra" ;set the inlude path to the correct place
;.DEVICE ATmega16 -- done in m16def.inc
.NOLIST
.INCLUDE "m16def.inc"
.LIST

.def TMP1=R16		;
.def TMP2=R17		;
.def TMP3=R18

.cseg
.org $000		;locate code at address $000
rjmp START		; Jump to the START Label

.org $02A		;locate code past the interupt vectors

START:
	ldi TMP1, LOW(RAMEND)	;initialise the stack pointer
	out SPL, TMP1
	ldi TMP1, HIGH(RAMEND)
	out SPH, TMP1
	call Init_LCD
	ldi r16, 0x41
	call Write_char
	ldi r16, 0x84
	call Write_instruc
	rcall delay
	ldi r16, 0x7e
	call Write_char
	ldi r16, 0x55
	call Write_char

MAIN_LOOP:
	NOP
	NOP
	RJMP MAIN_LOOP

.include "LCD.asm"
