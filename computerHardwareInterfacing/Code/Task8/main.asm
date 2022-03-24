; Portc0-7 connected to leds0-7
; ADC0 - VAR
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
.org ADCCaddr
rjmp ADC_ISR
.org $02A		;locate code past the interupt vectors

START: 	ldi TMP1, LOW(RAMEND)	;initialise the stack pointer
	out SPL, TMP1
	ldi TMP1, HIGH(RAMEND)
	out SPH, TMP1

;setup ADC in free running mode, with interrupt
	ldi tmp1, 0b01100000
	out ADMUX, tmp1
	ldi tmp1, 0xff
	out adcsra, tmp1

; portc output
	ldi tmp1, 0xff
	out ddrc, tmp1

;enable interrupts
	SEI

main_loop:
	nop
	nop
	rjmp main_loop

ADC_ISR:
	push tmp1
	in tmp1, ADCH
	out portc, tmp1
	pop tmp1
	RETI
