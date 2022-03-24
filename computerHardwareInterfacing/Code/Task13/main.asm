; Connect pb2 to led0
; pb3 (OC0) to led1
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
START: 	ldi TMP1, LOW(RAMEND)	;initialise the stack pointer
	out SPL, TMP1
	ldi TMP1, HIGH(RAMEND)
	out SPH, TMP1

	ldi tmp1, 0b00001100
	sbi PORTB, 2
	out ddrb, tmp1 ;set OC0 output
	ldi tmp1, 0xea
	out ocr0, tmp1	; set the compare value
	ldi tmp1, 0b01110001
	out tccr0, tmp1 ; set tcnt0 operation
main_loop:
	nop
	rjmp main_loop
