.INCLUDEPATH "/usr/share/avra"
.NOLIST
.INCLUDE "m16def.inc"
.LIST
.def bitpattern=R14
.def lightdelay=R15
.def OCRnum=R13
.def TMP1=R16		;
.def TMP2=R17		;
.def zero=R18
.def tasknum=r19
.dseg
RAMSTEPS: .BYTE 20
message2: .BYTE 20
flashylights: .byte 20
.eseg
EEMSG: .DB "EEP-hello world",0x00

.cseg
.org $000		;locate code at address $000
rjmp START		; Jump to the START Label
.org URXCaddr
rjmp URXC_ISR
.org UTXCaddr
rjmp UTXC_ISR
.org OC1Aaddr   ; locate next instruction at Timer0's overflow interrupt vector
jmp T1ocA_ISR
.org OVF0addr
jmp tc0ovf_isr
.org OVF2addr
jmp PWMset_ISR
.org OC2addr
jmp PWMclear_ISR
.org ADCCaddr
jmp ADCconversionISR
.org $02A		;locate code past the interupt vectors

START:
	ldi TMP1, LOW(RAMEND)	;initialise the stack pointer
	out SPL, TMP1
	ldi TMP1, HIGH(RAMEND)
	out SPH, TMP1
    ldi zero, 0x00
	call Init_UART
	call InitIO
	ldi tasknum, 0x00
	;read eep into ram
	    ldi XL, low(Message2)
    ldi XH, high(Message2)
    ldi YL, low(EEMSG)
    ldi YH, high(EEMSG)
    call READEE
    call Init_LCD
    call Initt5
    call task5
    ;ldi tmp1, 'A'
    ;call Write_char
    ;call task3
	call sendmenu
;	ldi zl, low(2*STEPCW)
;	ldi zh, high(2*STEPCW)
;	call READPGMMEM
;    LDI XL,low(RAMSTEPS)
;    LDI XH,high(RAMSTEPS)
 ;   ld tmp1, x
 ;   out portd, tmp1
	SEI
MAIN_LOOP:
	NOP
	NOP
	RJMP MAIN_LOOP
Initt5:
    LDI XL,low(flashylights)
    LDI XH,high(flashylights)
    ldi tmp1, 0x01
    st x+, tmp1
    ldi tmp1, 0x02
    st x+, tmp1
    ldi tmp1, 0x04
    st x+, tmp1
    ldi tmp1, 0x08
    st x+, tmp1
    ldi tmp1, 0x10
    st x+, tmp1
    ldi tmp1, 0x20
    st x+, tmp1
    ldi tmp1, 0x40
    st x+, tmp1
    ldi tmp1, 0x20
    st x+, tmp1
    ldi tmp1, 0x10
    st x+, tmp1
    ldi tmp1, 0x08
    st x+, tmp1
    ldi tmp1, 0x04
    st x+, tmp1
    ldi tmp1, 0x02
    st x+, tmp1
    ldi tmp1, 0x01
    st x+, tmp1
    ldi tmp1, 0x00
    st x+, tmp1
    ldi tmp1, 10
    mov OCRnum, tmp1
    out ocr2, OCRnum
ret


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

;---------------------------------------------
; jhfsdfdhfssfd
;------------------------------------------------
InitIO:
    ldi tmp1, 0xf0
    out DDRD, tmp1
	ret

READPGMMEM:
    LDI XL,low(RAMSTEPS)
    LDI XH,high(RAMSTEPS)
READPGM:
    LPM tmp1, Z+
    st X+, tmp1
    cpi TMP1, 0x00
    BRNE READPGM
    RET

URXC_ISR:
    in tmp1, udr
    cpi tmp1, '1'
    breq TASK1
    cpi tmp1, '2'
    breq TASK2
    cpi tmp1, '3'
    breq TASK3
    cpi tmp1, '4'
    breq TASK4
    reti

task1: ldi tasknum, 0x01
      	ldi zl, low(2*STEPCW)
	ldi zh, high(2*STEPCW)
	call READPGMMEM
    LDI XL,low(RAMSTEPS)
    LDI XH,high(RAMSTEPS)
    ld tmp1, x+
    out portd, tmp1
     ;sbi portd,7
     ldi tmp1,0x06
    out ocr1ah, tmp1
    ldi tmp1, 0x1b
    out OCR1AL, tmp1
    call initimer1
     reti

task2: ldi tasknum, 0x02
      	ldi zl, low(2*HSTEPCW)
	ldi zh, high(2*HSTEPCW)
	call READPGMMEM
    LDI XL,low(RAMSTEPS)
    LDI XH,high(RAMSTEPS)
    ld tmp1, x+
    out portd, tmp1
     ;sbi portd,7
     ldi tmp1,0x03
    out ocr1ah, tmp1
    ldi tmp1, 0x0d
    out OCR1AL, tmp1
    call initimer1
     reti

task3:
     ldi tasknum, 0x03
     ldi XL, low(Message2)
    ldi XH, high(Message2)
Message2LCD:      ld tmp1, x+
    cpi tmp1, 0x00
    breq m2ldone
    call Write_char
    rjmp Message2LCD
m2ldone:

     reti


task4:
sbi ddra, 0
sbi porta, 0
       ldi tasknum, 0x04
     ldi XL, low(Message2)
    ldi XH, high(Message2)
    ;ldi tmp1, 'D'
    ld tmp1, x+
    out UDR, tmp1
     reti

task5:
 ldi tmp1, 5
 mov lightdelay, tmp1
 ldi tmp1, 0xff
 out DDRA, tmp1
LDI XL,low(flashylights)
    LDI XH,high(flashylights)
   ldi tmp1, 0xC1
   out timsk, tmp1
   ldi tmp1, 0x05
   out tccr0, tmp1
   out tcnt2, zero
   ldi tmp1, 0x03
   out TCCR2, tmp1
   ldi tmp1, 0b01100111 ; AVCC selected as reference (limitation of wiring on board),
                        ; ADLAR set so that the most significant 8 bits are in ADCH read datasheet p207-225 and make sure
                        ; you are familiar with the ADC) and ADMUX set so that ADC0 is the input to the ADC
    OUT ADMUX, tmp1
    ldi tmp1, 0b11111111      ; ADC enabled, conversion started, no auto trigger, interrupt enabled and prescaler set to /128
    OUT ADCSRA, tmp1
   reti

UTXC_ISR:
  cpi tasknum, 0x04
  breq sendfromram
    lpm tmp1, Z+
    cpi tmp1, 0x00
    breq donetx
    out udr, tmp1
donetx:    reti




sendfromram:
 ; ldi tmp1,'X'
 ld tmp1, x+
 cpi tmp1, 0x00
breq txcexit
    out UDR, tmp1
txcexit:     reti

T1ocA_ISR:
sbi ddra,0
sbi porta,0
  ld tmp1, x+
  cpi tmp1, 0x00
  breq gotofirststep
doout:
  out portd, tmp1
  reti


gotofirststep:
    LDI XL,low(RAMSTEPS)
    LDI XH,high(RAMSTEPS)
    ld tmp1, x+
    rjmp doout

initimer1:
    ;out tcnt0, zero
     out tcnt1h, zero
     out tcnt1l, zero
    ldi tmp1, 0b00010000
    out timsk, tmp1
    ldi tmp1, 0b00001101
    out TCCR1B,tmp1
    ret

sendmenu:
   ldi zl, low(2*MENU)
	ldi zh, high(2*menu)
	lpm tmp1,Z+
	out UDR, tmp1
ret
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

tc0ovf_isr:
  dec lightdelay
  breq dolightstep
  reti
dolightstep:  ldi tmp1,5
  mov lightdelay, tmp1
  ld  tmp1, x+
  cpi tmp1, 0x00
  brne lightoutput
  LDI XL,low(flashylights)
    LDI XH,high(flashylights)
  ld  tmp1,x+
lightoutput:
  mov bitpattern, tmp1
  out porta, tmp1
reti

PWMCLEAR_ISR:
out ocr2, OCRnum
  out porta, zero
  reti

PWMSET_ISR:
  out porta,bitpattern
  reti
ADCconversionISR:
  in tmp1, ADCH
  mov ocrnum, tmp1
  reti

STEPCW: .DB 0x10,0x20,0x40,0x80,0x00,0x00
STEPCCW: .DB 0x80,0x40,0x20,0x10,0x00,0x00
HSTEPCW: .DB 0x10,0x30,0x20,0x60,0x40,0xc0,0x80,0x00
HSTEPCCW: .DB 0x80,0xc0,0x40,0x60,0x20,0x30,0x10,0x00

MENU: .DB 0x0c,"menu for tasks ",0x0d,0x0a
menu1: .db "1 task 1",0x0d,0x0a
menu2: .db "2 task 2",0x0d,0x0a
menu3: .db "3 task 3",0x0d,0x0a
menu4: .db "4 task 4",0x0d,0x0a
menu5: .db "5 task 5 ",0x0d,0x0a,0x00

blankterminal: .db 0x0c,0x00

.include "LCD.asm"
