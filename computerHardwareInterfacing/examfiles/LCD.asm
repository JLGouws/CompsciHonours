;                     LCD file
.MACRO LCD_WRITE
 CBI PORTB, 1
.ENDMACRO
.MACRO LCD_READ
 SBI PORTB, 1
.ENDMACRO
.MACRO LCD_E_HI
 SBI PORTB, 0
.ENDMACRO
.MACRO LCD_E_LO
 CBI PORTB, 0
.ENDMACRO
.MACRO LCD_RS_HI
 SBI PORTB, 2
.ENDMACRO
.MACRO LCD_RS_LO
 CBI PORTB, 2
.ENDMACRO

;This is a one millisecond delay
Delay:		
  push r16
  ldi r16, 11
Delayloop1: 	
  push r16
  ldi r16, 239 ; for an 8MHz xtal
Delayloop2:	dec r16
  brne Delayloop2
  pop r16
  dec r16
  brne Delayloop1
  pop r16
  ret
; waits 800 clock cycles (0.1ms on 8MHz clock)
Waittenth:	push r16
  ldi r16, 255
decloop:	dec r16
  nop
  nop
  brne decloop
  pop r16
  ret

; return when the lcd is not busy
Check_busy:	
  push r16
  ldi r16, 0b00000000
  out DDRC, r16	; portc lines input
  LCD_RS_LO	;RS lo
  LCD_READ	;read
Loop_Busy:	rcall Delay	; wait 1ms
  LCD_E_HI	; E hi
  rcall Delay
  in r16, PINC	; read portc
  LCD_E_LO	; make e low
  sbrc r16, 7	; check the busy flag in bit 7
  rjmp Loop_busy
  LCD_WRITE	;
  LCD_RS_LO	; rs lo
  pop r16
  ret

; write char in r16 to LCD
Write_char:	;rcall Check_busy
  push r17
  rcall Check_busy
  LCD_WRITE
  LCD_RS_HI
  ser r17
  out ddrc, r17	; c output
  out portc, R16
  LCD_E_HI
  LCD_E_LO
  clr r17
  out ddrc, r17
  ;rcall delay
  pop r17
  ret
;write instruction in r16 to LCD
Write_instruc:
  push r17
  rcall Check_busy
  LCD_WRITE
  LCD_RS_LO
  ser r17
  out ddrc, r17	; c output
  out portc, R16
  ;rcall delay
  LCD_E_HI
      LCD_E_LO
  clr r17
  out ddrc, r17
      ;rcall delay
  pop r17
  ret


Init_LCD:	push r16
  clr r16
  out ddrc, r16
  out portc, r16
  sbi ddrb, 2	;reg sel output
  sbi ddrb, 0	; enable output
  sbi portb, 2
  sbi portb, 0
  sbi ddrb, 1	; rw output
  ldi r16, 0x38
  rcall Write_instruc
  ldi r16, 0x0E ; turn lcd on with cursor
  rcall Write_instruc
  ldi r16, 0x06
  rcall Write_instruc
  ldi r16, 0x01
  rcall Write_instruc
  pop r16
  ret
