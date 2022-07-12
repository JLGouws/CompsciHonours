.INCLUDEPATH "/usr/share/avra"
.NOLIST
.INCLUDE "m16def.inc"
.LIST
.def TMP1=R16
.def TMP2=R2
.dseg
.org 0x00
MESSAGE: .db "Message 1" , 0x00
 .cseg
 .org $000
 rjmp START
 rjmp OC0_ISR

 .org INT_VECTORS_SIZE

 START: 
   ldi TMP1, LOW(RAMEND)
   out SPL , TMP1
   ldi TMP1, HIGH(RAMEND)
   out SPH, TMP1
   sbi DDRA, 0
   ldi tmp1 , 0x00
   out TCNT0, tmp1
   ldi tmp1, 0x02 
   out timsk , tmp1
   ldi tmp1 , 0x40
   out OCR0, tmp1
   ldi tmp1 , 0x01
   out TCCR0, tmp1
   SEI

 mainloop:
 nop
 nop
 rjmp mainloop

 OC0_ISR:
 in tmp1 , PORTA
 ld i tmp2 , 0x01
 eor tmp1 , tmp2
 out PORTA, tmp1
 RET
