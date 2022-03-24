; USART
; Connect RX and TX from usb-serial converter to TX and RX on board
; PB0 to led0
.INCLUDEPATH "/usr/share/avra" ;set the inlude path to the correct place
;.DEVICE ATmega16 -- done in m16def.inc
.NOLIST
.INCLUDE "m16def.inc"
.LIST

.def TMP1=R16		;
.def TMP2=R17		;
.def TMP3=R18
.def MESSAGE_offset=r19
.dseg
MESSAGE: .byte 10 ; reserve 10bytes for the message
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
	;set portb output for LEDs
	ser Tmp1
	out ddrb, Tmp1
	out portb, Tmp1
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
;increment message offset
	inc MESSAGE_OFFSET
;setup RAM pointer to the variable message
	LDI R30, low(MESSAGE)
	LDI R31, high(MESSAGE)
; increase by message offset.
	ADD R30, MESSAGE_OFFSET
	BRCC SEND_NEXT
	inc R31 ; there was an overflow, so increment the high byte
SEND_NEXT:
	LD Tmp1, Z
	cpi tmp1, 0x00
	breq message_finished
	out UDR, Tmp1
	RETI
message_finished:
	;reenable rx
	sbi UCSRB, RXCIE
	;clear led
	sbi portb, 0
	reti
; Code for RX complete
URXC_ISR:
	; led 0 off
	in tmp1, udr
	cbi portb, 0
	call SEND_MESSAGE
	RETI

SEND_MESSAGE:
	clr MESSAGE_OFFSET;
	LDI R30, low(MESSAGE)
	LDI R31, high(MESSAGE)
	LDI Tmp1, 'H'
	ST Z+, Tmp1
	LDI Tmp1, 'e'
	ST Z+, Tmp1
	LDI Tmp1, 'l'
	ST Z+, Tmp1
	LDI Tmp1, 'l'
	ST Z+, Tmp1
	LDI Tmp1, 'o'
	ST Z+, Tmp1
	LDI Tmp1, 0x00
	ST Z+, Tmp1
	LDI R30, low(MESSAGE)
	LDI R31, high(MESSAGE)
	LD Tmp1, Z
	out UDR, Tmp1 ; tx first char
	SBI UCSRB, TXCIE ; enable txci
	cbi UCSRB, RXCIE ; disable reception
	RET
