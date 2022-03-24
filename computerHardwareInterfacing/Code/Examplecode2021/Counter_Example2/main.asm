;Counter Example 2
; We are going to set up two LEDs to be pulse width modulated, one using the hardware
; Remember to patch PortA0 to an LED
; Patch PB3 (OC0) to an LED as well
; For you to do: After you have seen and understood the effects of changing the OCR value try writing code that will
;                increase the brightness of one led on each press of a button. (If you get this done then you have
;                essentially made a brightness control for a piece of display electronics.)

.INCLUDEPATH "/usr/share/avra" ;set the inlude path to the correct place
.NOLIST
.INCLUDE "m16def.inc"
.LIST

.def zero=R0        ; reserve R0 as our ZERO register
.def TMP1=R16		;defines serve the same purpose as in C,
.def TMP2=R17		;before assembly, defined values are substituted
.def TMP3=R18

.cseg			;Tell the assembler that everything below this is in the code segment
.org $000		;locate code at address $000
jmp START		; Jump to the START Label
.org OVF0addr   ; locate next instruction at Timer0's overflow interrupt vector
jmp T0ovf_ISR      ; Jump to the interrupt service routine
.org OC0addr   ; locate next instruction at Timer0's overflow interrupt vector
jmp T0oc_ISR      ; Jump to the interrupt service routine

.org $02A		;locate code past the interupt vectors
START:
  ldi TMP1, LOW(RAMEND)	;initialise the stack pointer
	out SPL, TMP1
	ldi TMP1, HIGH(RAMEND)
	out SPH, TMP1
	RCALL INITIALISE_IO ; Call the subroutine INITIALISE
	RCALL INIT_TIMER0
	; we have to set a value in OCR0
	ldi tmp1, 0xe8  ;Assemble with different values in here and make sure you can see the effect and explain it.
                    ; try 0x10, 0x40, 0x80, 0xb0, 0xd0
	out OCR0, tmp1
  SEI ; enable global interrupts
MAIN_LOOP:
	NOP
	NOP
	RJMP MAIN_LOOP

INITIALISE_IO:
	SBI DDRA, 0			; Make pin0 of portA output
  SBI DDRB, 3         ; set pb3 to be output
	CBI PORTA, 0		; Set the initial state of pin to low=LED-off
	CBI PORTB, 3		; Set the initial state of pin to low=LED-off
	RET					;Return from subroutine

; We are going to start with something simple, we will initialise timer0 with a prescale value
; of /64.
; TIMSK=0b00000011 (this will enable the overflow interrupt and output compare interrupt for T0)
; TCCR0=0b00000101 (all other functionality ignored, just setting the prescaler to /64)
INIT_TIMER0:
  ldi tmp1, 0b00000011
  out TIMSK, tmp1 ;Timer0 overflow interrupt is now enabled
  ldi tmp1, 0b01111011 ; FAST PWM inverting - read the datasheet and your notes
  out TCCR0, tmp1 ;Timer0 is now counting up with a prescaler of /64
  RET

T0oc_ISR:
  cbi PORTA, 0
  RETI

; To be pedantically correct we should be preseving the sreg, tmp1 and tmp2 on the stack, but I am tired of typing,
; It will not affect us for now . . . .
T0ovf_ISR:
  sbi PORTA,0
  RETI
