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

.def TMP1=R16		;
.def TMP2=R17		;
.dseg

.eseg

.cseg
.org $000		;locate code at address $000
rjmp START		; Jump to the START Label
.org URXCaddr
rjmp URXC_ISR
.org UDREaddr
rjmp UDRE_ISR
.org UTXCaddr
rjmp UTXC_ISR

.org $02A		;locate code past the interupt vectors

START:
	ldi TMP1, LOW(RAMEND)	;initialise the stack pointer
	out SPL, TMP1
	ldi TMP1, HIGH(RAMEND)
	out SPH, TMP1
	call Init_UART
	ldi tmp1, 0x55 ; Send a 'U' on startup - to see this connect with putty and press the reset button.
	out UDR, tmp1
	SEI
MAIN_LOOP:
	NOP
	NOP
	RJMP MAIN_LOOP

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
    inc tmp1
    out UDR, tmp1
    RETI
