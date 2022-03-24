; LCD_Example1
;Portc0-7 connected to LCD_data0-7
;PB0 - E
;PB1 - rw
;PB2 - rs
; TODO: Alter the address location that 'Bye' starts at so that it is present on the second line.
;       *Make sure you are happy with this, after learning about storing strings in
;        eeprom and program memory we will re-visit this.
;       * Go on . . . do it . . . make it display 'Hello World'
.INCLUDEPATH "/usr/share/avra" ;set the inlude path to the correct place
;.DEVICE ATmega16 -- done in m16def.inc
.NOLIST
.INCLUDE "m16def.inc"
.LIST

.def TMP1=R16		;
.def TMP2=R17		;

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
	ldi r16, 0x48
	call Write_char
	ldi r16, 0x65
	call Write_char
	ldi r16, 0x6c
	call Write_char
	call Write_char
	ldi r16, 0x6f
	call Write_char

	ldi r16, ' '
	call Write_char
	ldi r16, 'W'
	call Write_char
	ldi r16, 'o'
	call Write_char
	ldi r16, 'r'
	call Write_char
	ldi r16, 'l'
	call Write_char
	ldi r16, 'd'
	call Write_char
	ldi r16, '!'
	call Write_char

	ldi r16, 0x8d
	call Write_instruc
	rcall delay

MAIN_LOOP:
	NOP
	NOP
	RJMP MAIN_LOOP

.include "LCD.asm"
