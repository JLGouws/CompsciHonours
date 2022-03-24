;ADC Example 1
; Hardware connections.
; Some LEDS to show the results of our conversion
;PC0-led0
;PC1-led1
;PC2-led2
;PC3-led3
;PC4-led4
;PC5-led5
;PC6-led6
;PC7-led7
;To trigger our conversion
;PD2-sw0
;PA0 to VAR , the output of our potentiometer so we can vary the voltage to digitise

;----------------------------------------------------------------------------------------
; todo:  Turn the pot and trigger conversions, note the binary output of the conversion.
; * Change the code so that ADLAR is not used.  Read in both ADL & ADH and format the output
; so that it is the same as before.
; * Change your code again so that the lower 8 bits are now displayed.  Note that sometimes (when the potentiometer
;   is not moved you will get different results - the AD conversion is not always exact.
;   Change the prescaler settings to make the conversion more inaccurate.)

.INCLUDEPATH "/usr/share/avra" ;set the inlude path to the correct place

.NOLIST
.INCLUDE "m16def.inc"
.LIST

.def zero=R0        ; reserve R0 as our ZERO register
.def TMP1=R16		;defines serve the same purpose as in C,
.def TMP2=R17		;before assembly, defined values are substituted

.cseg			;Tell the assembler that everything below this is in the code segment
.org $000		;locate code at address $000
jmp START		; Jump to the START Label
.org INT0addr
jmp int0isr
.org ADCCaddr   ; locate next instruction at Timer0's overflow interrupt vector
jmp ADC_ISR      ; Jump to the interrupt service routine

.org $02A		;locate code past the interupt vectors
START:
    ldi TMP1, LOW(RAMEND)	;initialise the stack pointer
	out SPL, TMP1
	ldi TMP1, HIGH(RAMEND)
	out SPH, TMP1
	RCALL INITIALISE_IO ; Call the subroutine INITIALISE
    SEI ; enable global interrupts
MAIN_LOOP:
	NOP
	NOP
	RJMP MAIN_LOOP


INITIALISE_IO:
    ldi tmp1, 0xFF; Set porta as output to show the results of our conversion
    out DDRC, tmp1;
    sbi portd, 2 ; pullup on pd2 pin for int 0
    ldi tmp1, 0b01000000
    out GICR, tmp1 ; int0 enabled
    ldi tmp1, 0b00000010
    out mcucr, tmp1 ;falling edge triggered.
RET					;Return from subroutine

ADC_ISR:
 ;output conversion result
 in tmp1, ADCH
 out portc, tmp1
 ;Clear int0 flags
 ldi tmp1, 0b01000000
 out GIFR, tmp1 ; int0 enabled
 ;re-enable int0
 out GICR, tmp1 ; int0 enabled
    RETI

int0isr:
;SBI portc, 0
call Start1conversion
out gicr, zero ; diable external interrupts
   RETI

Start1conversion:
    ldi tmp1, 0b01100000 ; AVCC selected as reference (limitation of wiring on board),
                        ; ADLAR set so that the most significant 8 bits are in ADCH read datasheet p207-225 and make sure
                        ; you are familiar with the ADC) and ADMUX set so that ADC0 is the input to the ADC
    OUT ADMUX, tmp1
    ldi tmp1, 0b11001111      ; ADC enabled, conversion started, no auto trigger, interrupt enabled and prescaler set to /128
    OUT ADCSRA, tmp1
     ; so the clock to the adc is set to 8MHz/128=62.5kHz
     ;  The first conversion will take 25 clock cycles (see page 211 of data sheet) so the conversion will take 0.4ms
     RET
