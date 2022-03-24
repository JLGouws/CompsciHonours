; USART
; Connect RX and TX from usb-serial converter to TX and RX on board
; The current code just sets up the UART to RX and TX at 9600,
; when a character is received, it is "incremented" and then sent back - mainly to avoid
; any problems with "local echo" settings that may or may not be set in putty.
; Remember after programming the device, change the mode jumper - unplug usb - replug usb and then open putty
.INCLUDEPATH "/usr/share/avra"
.NOLIST
.INCLUDE "m16def.inc"
.LIST

.def zero=R2 		;
.def step=R3 		;
.def TMP1=R16		;
.def TMP2=R17		;
.def stepCount=R18		;
.def maxStepCount=R19		;
.def slow=R20

.dseg
STEP_TABLE: .byte 16

.eseg
EEM_STEP_TABLE: .DB 0x01, 0x30, 0x20, 0x60, 0x40, 0xC0, 0x80, 0x90, 0x00

.cseg
.org $000		;locate code at address $000
rjmp START		; Jump to the START Label
.org URXCaddr
jmp URXC_ISR
.org UDREaddr
jmp UDRE_ISR
.org UTXCaddr
jmp UTXC_ISR
.org OC0addr
jmp T0OC_ISR

.org $02A		;locate code past the interupt vectors

START:
	ldi TMP1, HIGH(RAMEND)
	out SPH, TMP1
	ldi TMP1, LOW(RAMEND)	;initialise the stack pointer
	out SPL, TMP1
  eor zero, zero
  ldi stepCount, 0x00;
  ldi maxStepCount, 0x00 
  CALL READ_STEP_TABLE
	call Init_UART
  CALL INITIALISE_IO
  CALL Initialise_t0
	SEI


MAIN_LOOP:
	NOP
	NOP
	RJMP MAIN_LOOP

INITIALISE_IO:
    ldi tmp1, 0xF0;
    out DDRD, tmp1;
    ldi tmp1, 0x01; 1 because we want to be past the null on the boundary -
    mov step, tmp1
    out GICR, tmp1 ; int0 enabled
    ldi tmp1, 0b00000010
    out mcucr, tmp1 ;falling edge triggered.
    RET					;Return from subroutine

Initialise_t0:
  ldi tmp1, 0xB0 ;
  out ocr0, tmp1 ; setup oc0
  ldi tmp1, 0b00000010
  out timsk, tmp1 ; enable t0 ocint - we use the output compare intterupt in this case
                  ; because we will set the timer to CTC mode
  ret

READ_STEP_TABLE:
  LDI XL, LOW(STEP_TABLE)
  LDI XH, HIGH(STEP_TABLE)
  LDI YL, LOW(EEM_STEP_TABLE)
  LDI YH, HIGH(EEM_STEP_TABLE)
  ;Read in initial null
  OUT EEARH, YH
  OUT EEARL, YL
  SBI EECR, EERE
  IN TMP1, EEDR
  ST X+, TMP1
  ADIW YL, 1
cStepRead:
  OUT EEARH, YH
  OUT EEARL, YL
  SBI EECR, EERE
  IN TMP1, EEDR
  ST X+, TMP1
  ADIW YL, 1
  CPI TMP1, 0x00
  BRNE cStepRead
  LDI TMP1, 0x00 ;null byte for end of step table
  ST X+, TMP1
  RET

Init_UART:
;set baud rate (9600,8,n,2)
 	ldi Tmp1, 51
	ldi Tmp2, 0x00
	out UBRRH, Tmp2
	out	UBRRL, Tmp1
;set rx and tx enable
	sbi UCSRB, RXEN
	sbi UCSRB, TXEN
; enable uart interrupts
	sbi UCSRB, RXCIE
	RET

; Interrupt code for when UDR empty
UDRE_ISR:
	RETI
; Code for TX complete
UTXC_ISR:
	RETI
; Code for RX complete
URXC_ISR:
  in tmp1, UDR
  cpi tmp1, 13
  breq startSteps
  out udr, tmp1;echo to terminal
  ldi tmp2, 10
  mul maxStepCount, tmp2
  mov maxStepCount, R0
  subi tmp1, '0'
  add maxStepCount, tmp1
  RETI
startSteps:
  ldi tmp1, 0b00001101; ctc mode, no OC pin use, /1024 prescaler
  out TCCR0, tmp1; start timer 0
  RETI

T0OC_ISR:
  inc slow
  cpi slow, 0x10
  brne contSteps
  mov slow, zero
  call Nextstep
  cp stepCount, maxStepCount
  brne contSteps 
  ldi tmp1, 0; ctc mode, no OC pin use, /1024 prescaler
  out TCCR0, tmp1; stop timer 0
  ldi stepCount, 0x00;
  ldi maxStepCount, 0x00;
  ldi slow, 0xB
contSteps:
  RETI

NEXTSTEP:
  inc step ; get to the next step
  inc stepCount ; get to the next step
; get correct value from lookuptable
getstep:
  ldi xl, low(STEP_TABLE)
  ldi xh, high(STEP_TABLE)
  add xl, step
  adc xh, zero
  ld tmp2, X
  cpi tmp2, 0x00
  brne outstep
 ; if we got a null then we need to wrap around
  ldi tmp2, 0x01 ;using tmp2 since we need the value in tmp1 and tmp2's value is rubbish at this moment'
  mov step, tmp2
  rjmp getstep
outstep:
  IN tmp1, PORTD
  andi tmp1, 0x0F ;tmp1 now contains the masked off values of portD
  or tmp1, tmp2
  out portd, tmp1
  RET
