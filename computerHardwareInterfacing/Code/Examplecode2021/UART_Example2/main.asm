; USART
; Connect RX and TX from usb-serial converter to TX and RX on board

.INCLUDEPATH "/usr/share/avra"
.NOLIST
.INCLUDE "m16def.inc"
.LIST

.def TMP1=R16		;
.def TMP2=R17		;
.dseg
BUFFER: .byte 10
.eseg
MSG: .db "Hello",0x00
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
	; Read a message from eeprom into the buffer
	ldi XL, low(Buffer)
    ldi XH, high(Buffer)
    ldi YL, low(MSG)
    ldi YH, high(MSG)
    call READEE
    ; init uart
	call Init_UART
	SEI
MAIN_LOOP:
	NOP
	NOP
	RJMP MAIN_LOOP

READEE:
    OUT EEARH, YH
    OUT EEARL, YL
    SBI EECR, EERE
    IN TMP1, EEDR
    st X+, tmp1
    adiw YL, 1
    cpi TMP1, 0x00
    BRNE READEE
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
	sbi UCSRB, TXCIE
	RET

; Interrupt code for when UDR empty
UDRE_ISR:
	RETI
; Code for TX complete
UTXC_ISR:
    ;get the next byte and post increment X
    ld tmp1, x+
    cpi tmp1, 0x00
    breq gotnull ;if the nex char to send is a null don't send it
    ; Send it out into the world
    out udr, tmp1
donetx:
	RETI

gotnull:

    RETI
; Code for RX complete
URXC_ISR:
    in tmp1, udr; This looks pointless doesn't it?
    ; Comment it out and see if you can explain the behaviour - read through the reception section in the datasheet
    ; when we get a received char say hello - it is only polite
    call send_buffer
    RETI

Send_buffer:
    ; point X to beginning of buffer
    ldi XL, low(Buffer)
    ldi XH, high(Buffer)
    ;get the first byte and post increment X
    ld tmp1, x+
    ; Send it out into the world
    out udr, tmp1
        RET
