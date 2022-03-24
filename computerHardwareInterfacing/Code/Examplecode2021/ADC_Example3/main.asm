;ADC Example 3
; Hardware connections.
; Some LEDS to show the results of our conversion for debugging
;PC0-led0
;PC1-led1
;PC2-led2
;PC3-led3
;PC4-led4
;PC5-led5
;PC6-led6
;PC7-led7

;PD7-DRV3
;PD6-DRV2
;PD5-DRV1
;PD4-DRV0


;PA0 to VAR , the output of our potentiometer so we can vary the voltage to digitise

;----------------------------------------------------------------------------------------
; todo:  NOTE if you try to drive the motor too fast, you will just get a high-pitched whine from it
;       * Can you alter the code so that the speed of the morot only updates when int0 is pressed
;         (there is an easy way and an easier way)

.INCLUDEPATH "/usr/share/avra" ;set the inlude path to the correct place

.NOLIST
.INCLUDE "m16def.inc"
.LIST

.def zero=R0        ; reserve R0 as our ZERO register
.def step=R1
.def TMP1=R16		;defines serve the same purpose as in C,
.def TMP2=R17		;before assembly, defined values are substituted

.cseg			;Tell the assembler that everything below this is in the code segment
.org $000		;locate code at address $000
jmp START		; Jump to the START Label
.org OC0addr
jmp OC0_isr
.org ADCCaddr   ; locate next instruction at Timer0's overflow interrupt vector
jmp ADC_ISR      ; Jump to the interrupt service routine

.org $02A		;locate code past the interupt vectors
START:
    ldi TMP1, LOW(RAMEND)	;initialise the stack pointer
	out SPL, TMP1
	ldi TMP1, HIGH(RAMEND)
	out SPH, TMP1
	RCALL INITIALISE_IO ; Call the subroutine INITIALISE
	call initialise_t0 ; call subroutine to init t0 with output comparevalue
    call Startfreerunningconversion
    SEI ; enable global interrupts
MAIN_LOOP:
	NOP
	NOP
	RJMP MAIN_LOOP


INITIALISE_IO:
    ldi tmp1, 0xF0; Set porta as output to show the results of our conversion
    out DDRD, tmp1;
    sbi portd, 2 ; pullup on pd2 pin for int 0
    ldi tmp1, 0x10
    mov step, tmp1
    ldi tmp1, 0b01000000
    out GICR, tmp1 ; int0 enabled
    ldi tmp1, 0b00000010
    out mcucr, tmp1 ;falling edge triggered.
RET					;Return from subroutine

Initialise_t0:
    ldi tmp1, 0xff ; try a range of values here and observe the motor speed. As an exercise determine the # of steps per
                    ; second that the motor is performing for each value
    out ocr0, tmp1 ; setup oc0
    ldi tmp1, 0b00000010
    out timsk, tmp1 ; enable t0 ocint - we use the output compare intterupt in this case
                    ; because we will set the timer to CTC mode
    ldi tmp1, 0b00001101; ctc mode, no OC pin use, /1024 prescaler
    out TCCR0, tmp1
    ret

; On overflow of timer we reinit int0, after clearing the flags, stop the timer.
ADC_ISR:
;our conversion result goes into OC0
 in tmp1, adch
 out ocr0, tmp1
    RETI

OC0_isr:
   CALL NEXTSTEP
   RETI

Startfreerunningconversion:
    ldi tmp1, 0b01100000 ; AVCC selected as reference (limitation of wiring on board),
                        ; ADLAR set so that the most significant 8 bits are in ADCH read datasheet p207-225 and make sure
                        ; you are familiar with the ADC) and ADMUX set so that ADC0 is the input to the ADC
    OUT ADMUX, tmp1
    ldi tmp1, 0b01100000 ;free running mode to trigger from T0 compare match
    out SFIOR, tmp1
    ldi tmp1, 0b11101111      ; ADC enabled, conversion started, no auto trigger, interrupt enabled and prescaler set to /128
    OUT ADCSRA, tmp1
     RET
NEXTSTEP:
IN tmp1, PORTD
andi tmp1, 0x0F ;tmp now contains the masked off values of portD

lsl step ; increment out variable
brne outstep
ldi tmp2, 0x10
mov step, tmp2

outstep:
  or tmp1, step
  out portd, tmp1
  RET

