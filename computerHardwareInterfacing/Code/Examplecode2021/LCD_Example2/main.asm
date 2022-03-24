.INCLUDEPATH "/usr/share/avra"
.NOLIST
.INCLUDE "m16def.inc"
.LIST

.def TMP1=R16		;
.def TMP2=R17		;

.eseg
EEMSG: .DB "EEP-hello world",0x00

.dseg
message1: .BYTE 20
message2: .BYTE 20
.cseg
.org $000
rjmp START

.org $02A
START:
	ldi TMP1, LOW(RAMEND)
	out SPL, TMP1
	ldi TMP1, HIGH(RAMEND)
	out SPH, TMP1
	call Init_LCD
    ldi XL, low(Message1)
    ldi XH, high(Message1)
    ldi YL, low(EEMSG)
    ldi YH, high(EEMSG)
    call READEE
    ldi XL, low(Message2)
    ldi XH, high(Message2)
    ldi ZL, low(2*PGMMSG)
    ldi ZH, high(2*PGMMSG)
    call READPGM
    ldi XL, low(Message2)
    ldi XH, high(Message2)
    call Message2LCD

MAIN_LOOP:
	NOP
	NOP
	RJMP MAIN_LOOP

Message2LCD:
    ldi tmp2, 0x00
oneChar:
    cpi tmp2, 16
    brne WriteChar
    ldi tmp1, 0b11000000 
    call Write_instruc
WriteChar:
    ld tmp1, x+
    cpi tmp1, 0x00
    breq m2ldone
    call Write_char
    inc tmp2
    rjmp oneChar
m2ldone:
    RET

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

READPGM:
    LPM tmp1, Z+
    st X+, tmp1
    cpi TMP1, 0x00
    BRNE READPGM
    RET

.include "LCD.asm"
PGMMSG: .db "PM-hello world, this is a message",0x00
