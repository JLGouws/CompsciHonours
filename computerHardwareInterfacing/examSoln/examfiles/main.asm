.INCLUDEPATH "/usr/share/avra" ;set the inlude path to the correct place
;.DEVICE ATmega16
.NOLIST
.INCLUDE "m16def.inc"
.LIST
.def TMP1=R16		;
.def TMP2=R17		;
.def TMP3=R18

.cseg			;Tell the assembler that everything below this is in the code segment
.org 0x000		;locate code at address $000
jmp START		; Jump to the START Label
.org INT0addr
jmp INT0_ISR
.org INT1addr
jmp INT1_ISR
.org $02A		;locate code past the interupt vectors

START:
	ldi TMP1, LOW(RAMEND)	;initialise the stack pointer
	out SPL, TMP1
	ldi TMP1, HIGH(RAMEND)
	out SPH, TMP1
	call init_IO
	CALL Init_stepper
	call Init_LCD
	SEI
MAIN_LOOP:
	NOP
	NOP
	NOP
	RJMP MAIN_LOOP

INT0_ISR:

  reti

INT1_ISR:
  reti

.include "LCD.asm"
menu_text:   .db 0x0c,"Project Tasks: ",0x0d,0x0a
menu_text1:  .db "--------------",0x0d,0x0a
menu_text2:  .db "1) Rotate clockwise for 5 seconds ",0x0d,0x0a
menu_text3:  .db "2) Rotate anti-clockwise for 5 seconds",0x0d,0x0a
menu_text4:  .db "3) Rotate clockwise for 10 seconds",0x0d,0x0a
menu_text5:  .db "4) Rotate anti-clockwise for 10 seconds ",0x0d,0x0a
menu_text6:  .db "5) Disable stepper",0x0d,0x0a
menu_text7:  .db "6) Enable stepper ",0x0d,0x0a
menu_text8:  .db "7) Start ADC PWM task ",0x0d,0x0a
menu_text9:  .db "8) Stop ADC PWM task",0x0d,0x0a
menu_text10: .db "9) Print 3rd stored message to LCD",0x0d,0x0a
menu_text11: .db "10) Clear LCD ",0x0d,0x0a
menu_text12: .db "11) Reset Microcontroller ",0x0d,0x0a,0x00,0x00
blankterminal: .db 0x0c,0x00
SINGLE_STEPS: .db 0x00,0x10,0x20,0x40,0x80,0x00
HALF_STEPS:   .db 0x00,0x10,0x20,0x40,0x80,0x00

