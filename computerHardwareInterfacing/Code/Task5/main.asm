; wire switches 0-1 to portd0-1
; wire leds0-1 to porta0-1
.INCLUDEPATH "/usr/share/avra" ;set the inlude path to the correct place
;.DEVICE ATmega16 -- done in m16def.inc
.NOLIST
.INCLUDE "m16def.inc"
.LIST
.def TMP1=R16		;
.def TMP2=R17		;
.def TMP3=R18
.def HCOUNT=R19
.def SECCOUNT=R20

.cseg
.org $000		;locate code at address $000
rjmp START		; Jump to the START Label
.org INT0addr
rjmp INT0_ISR
.org INT1addr
rjmp INT1_ISR
.org OC1Baddr
rjmp TIMER1_OC_ISR
.org $02A		;locate code past the interupt vectors

START:
	ldi TMP1, LOW(RAMEND)	;initialise the stack pointer
	out SPL, TMP1
	ldi TMP1, HIGH(RAMEND)
	out SPH, TMP1

	RCALL INITIALISE_PORTS
	RCALL INITIALISE_TIMER
	RCALL INITIALISE_EXTERNAL_INTERRUPTS
	SEI
MAIN_LOOP:
	NOP
	NOP
	RJMP MAIN_LOOP

INITIALISE_PORTS:
	; portb output LEDs
	LDI TMP1, 0x00
	out PORTA, tmp1
	LDI TMP1, 0xFF
	OUT DDRA, TMP1
	;portd inputs
	LDI TMP1, 0x00
	OUT DDRD, TMP1
	;enable pullups on int pins
	sbi PORTD, 2
	sbi PORTD, 3
	RET

INITIALISE_TIMER:
	;enable timer 1 interrupt
	clr tmp1
	ldi tmp1, 0b00001000
	out TIMSK, tmp1
	clr tmp1
	;set initial value in timer
	out TCNT1H, tmp1
	out TCNT1L, tmp1
  ;set the overflow registers
  ldi tmp1, 0x00
  ldi tmp2, 78
  out OCR1AH, tmp1
  out OCR1AL, tmp2
  ldi tmp1, 0b00000101;prescaler for time
  out TCCR1B, tmp1
	RET

INITIALISE_EXTERNAL_INTERRUPTS:
	;enable  int0
	ldi tmp1, 0x40
	out GICR, tmp1
	; interrupt on falling edge.
	ldi tmp1, 0x0a
	out MCUCR, tmp1
	ret

INT0_ISR:
	;wait a while and then light the test led
;disable int 0 and enable int1
	ldi tmp1, 0x80
	out GICR, tmp1
	sbi porta,0
    ldi tmp1, 0x60
loop1: ser tmp2
loop2: ser tmp3
loop3: ;
	dec tmp3
	cpi tmp3, 0
	BRNE loop3
	dec tmp2
	BRNE loop2
	dec tmp1
	BRNE loop1
	;we have now finished a bit of a delay.
	sbi porta, 1 ;turn led on
	cbi porta,0
	;start timer going
	ldi tmp1, 0x05
	out tccr1b, tmp1
	reti

INT1_ISR:
; stop timer1
	ldi tmp1, 0x00
	out tccr1b, tmp1
;disable int 1 and enable int0
	ldi tmp1, 0x40
	out GICR, tmp1
;read both timer bytes and display (we will only display the high byte}
	out PORTA, SECCOUNT
  sbi Porta, 7
	reti

TIMER1_OC_ISR:
    inc HCOUNT
    CPI HCOUNT, 100
    BREQ TOGGLE
    RJMP FINISH
  TOGGLE:
    INC SECCOUNT
    LDI HCOUNT, 0x00
  FINISH:
    ldi tmp1, 0x00
    out TCNT1H, tmp1
    out TCNT1L, tmp1
    ldi tmp1, 0b00000101;prescaler for time
    out TCCR1B, tmp1
    RETI
