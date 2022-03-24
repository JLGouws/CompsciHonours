;Counter Example 1
;
; Remember to patch PortA0 to an LED
; For you to do:
;    - Setup T0 to generate an Ouput compare interrupt 0.01s after starting (use an output compare interrupt)
;    - in your isr do not forget to stop, zero and restart your timer
;    - increment a register, when it gets to 100 toggle the LED, the LED should now be on for 1sec and off for 1sec

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
jmp T0_ISR      ; Jump to the interrupt service routine
.org $02A		;locate code past the interupt vectors
START:
    ldi TMP1, LOW(RAMEND)	;initialise the stack pointer
	out SPL, TMP1
	ldi TMP1, HIGH(RAMEND)
	out SPH, TMP1
	RCALL INITIALISE_IO ; Call the subroutine INITIALISE
	RCALL INIT_TIMER0
    SEI ; enable global interrupts
MAIN_LOOP:
	NOP
	NOP
	RJMP MAIN_LOOP


INITIALISE_IO:
	SBI DDRA, 0			; Make pin0 of portA output
	CBI PORTB, 0		; Set the initial state of pin to low=LED-off
	RET					;Return from subroutine

; We are going to start with something simple, we will initialise timer0 with the highest possible prescale value
; which is /1024.  This means that the effective count rate will be ~8000 counts per second, which means that the
; counter will overflow about 31 times per second, too fast to see, but bear with me.
; So before we start we need to know what 'arcane values' to put in the gpio registers that control timer0
; TIMSK=0b00000001 (this will enable the overflow interrupt for T0)
; TCCR0=0b00000101 (all other functionality ignored, just setting the prescaler to /1024)
INIT_TIMER0:
    ldi tmp1, 0b00000001
    out TIMSK, tmp1 ;Timer0 overflow interrupt is now enabled
    ldi tmp1, 0b00000101
    out TCCR0, tmp1 ;Timer0 is now counting up with a prescaler of /1024
    RET

; To be pedantically correct we should be preseving the sreg, tmp1 and tmp2 on the stack, but I am tired of typing,
; It will not affect us for now . . . .
T0_ISR:
    in tmp1, porta ; read in the values that are currently being output in porta
                  ; we do not want to read the pin values since then we would have to be carefull
                  ; with masking values as we may enable/disable pullups
    ldi tmp2, 0b00000001
    eor tmp1, tmp2 ; toggle bit 0
    out porta, tmp1 ; output back to porta
    RETI
