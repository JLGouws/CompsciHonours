; USART
; Connect RX and TX from usb-serial converter to TX and RX on board

.INCLUDEPATH "/usr/share/avra"
.NOLIST
.INCLUDE "m16def.inc"
.LIST
.def zero=r0
.def TMP1=R16		;
.def TMP2=R17		;
.def rxcnt=r18
.dseg
BUFFER: .byte 20
.cseg
.org $000		;locate code at address $000
rjmp START		; Jump to the START Label
.org URXCaddr
rjmp URXC_ISR

.org $02A		;locate code past the interupt vectors

START:
	ldi TMP1, LOW(RAMEND)	;initialise the stack pointer
	out SPL, TMP1
	ldi TMP1, HIGH(RAMEND)
	out SPH, TMP1
    ; init uart
    ldi tmp1, 0x00
    mov zero, tmp1
    mov rxcnt, tmp1
	call Init_UART
    call Init_LCD

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
	sbi portd, 1 ; enable pullup on unused transmitter pin - see what happens when you remove this line
; enable uart interrupts
	sbi UCSRB, RXCIE
	RET

; Code for RX complete
URXC_ISR:
    in tmp1, udr
    cpi tmp1, '*'
    breq Processbuffer
    cpi rxcnt, 0x0F  ; cpi is rd-K
    BREQ overrun
    ; if we get here we have not got a buffer overrun
    inc rxcnt
    out porta, rxcnt
    ldi xl, low(BUFFER)
    ldi xh, high(BUFFER)
    ;add offset
    add xl, rxcnt
    adc xh, zero
    st x, tmp1
    ld tmp1, x
    call Write_char
        RETI

Processbuffer:
    sbi ddra, 7
    sbi porta, 7
    mov rxcnt, zero
    RETI

overrun:
    sbi ddra, 6
    sbi porta, 6
    mov rxcnt, zero
    reti
.include "LCD.asm"
