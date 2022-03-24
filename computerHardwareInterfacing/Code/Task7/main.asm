; Stepper Motor 1
; PD2 - SW0
; PD3 - SW1
; PD4 - DRV0
; ..
; PD7 - DRV3
; Also make sure that the jumper by the motor
; Connection is set to 2-3 (+5v)
.INCLUDEPATH "/usr/share/avra" ;set the inlude path to the correct place
;.DEVICE ATmega16 -- done in m16def.inc
.NOLIST
.INCLUDE "m16def.inc"
.LIST
.def TMP1=R16		;
.def TMP2=R17		;
.def TMP3=R18
.def mask=R19
.cseg
.org $000		;locate code at address $000
rjmp START		; Jump to the START Label
.org INT0addr
rjmp INT0_ISR
.org INT1addr
rjmp INT1_ISR
.org OVF0addr
rjmp T0_OVF_ISR
.org $02A		;locate code past the interupt vectors

START:
	ldi TMP1, LOW(RAMEND)	;initialise the stack pointer
	out SPL, TMP1
	ldi TMP1, HIGH(RAMEND)
	out SPH, TMP1

  ; Set PORT D to correct IO
  ldi TMP1, 0b11110000
  OUT DDRD, TMP1
  ; Set initial state on port D
  LDI TMP1, 0b00001100
  OUT PORTD, TMP1
  ; setup timer 1 for overflow
  LDI TMP1, 0x00
  out TCNT0, tmp1
  ldi tmp1, 0x01
  out TIMSK, tmp1
  ldi tmp1, 0x05
  out TCCR0, tmp1
  ldi MASK, 0b00010000
  out portd, mask
  SEI

loop:
  nop
  nop
  nop
  rjmp loop

INT0_ISR:
	RETI

INT1_ISR:
	RETI

T0_OVF_ISR:
	; single step through pd4-7
    LSL MASK
;inc mask
    BREQ SET_PD4
    RJMP do_out
set_pd4:
    LDI MASK, 0b00010000
do_out:
    in TMP1, PORTD;
	andi tmp1, 0x0f
	or tmp1, mask
	out PORTD, mask
	RETI
