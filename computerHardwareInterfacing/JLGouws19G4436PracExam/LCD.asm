;                     LCD file
; Handles the LCD screen
; J L Gouws and Mr. Sullivan
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
  PUSH  r16
  LDI   r16, 11
Delayloop1: 	
  PUSH  r16
  LDI   r16, 239                  ; for an 8MHz xtal
Delayloop2:	
  DEC   r16
  BRNE  Delayloop2
  POP   r16
  DEC   r16
  BRNE  Delayloop1
  POP   r16
  RET
; waits 800 clock cycles (0.1ms on 8MHz clock)
Waittenth:	
  PUSH  r16
  LDI   r16, 255
decloop:	
  DEC   r16
  NOP
  NOP
  BRNE  decloop
  POP   r16
  RET

; return when the lcd is not busy
Check_busy:	
  PUSH  r16
  LDI   r16, 0b00000000
  OUT   DDRC, r16	                ; portc lines input
  LCD_RS_LO	                      ; RS lo
  LCD_READ	                      ; read
Loop_Busy:	
  RCALL Delay	                    ; wait 1ms
  LCD_E_HI	                      ; E hi
  RCALL Delay
  IN    r16, PINC	                ; read portc
  LCD_E_LO	                      ; make e low
  SBRC  r16, 7	                  ; check the busy flag in bit 7
  RJMP  Loop_busy
  LCD_WRITE	                      ;
  LCD_RS_LO	                      ; rs lo
  POP   r16
  RET

; write char in r16 to LCD
Write_char:	                      ;rcall Check_busy
  PUSH  r17
  RCALL Check_busy
  LCD_WRITE
  LCD_RS_HI
  SER   r17
  OUT   DDRC, r17	                ; c output
  OUT   PORTC, R16
  LCD_E_HI
  LCD_E_LO
  CLR   r17
  OUT   DDRC, r17
  POP   r17
  RET
;write instruction in r16 to LCD
Write_instruc:
  PUSH  r17
  RCALL Check_busy
  LCD_WRITE
  LCD_RS_LO
  SER   r17
  OUT   DDRC, r17	                ; c output
  OUT   PORTC, R16
  LCD_E_HI
  LCD_E_LO
  CLR   r17
  OUT   DDRC, r17
  POP   r17
  RET


Init_LCD:	
  PUSH  r16
  CLR   r16
  OUT   DDRC, r16
  OUT   PORTC, r16
  SBI   DDRB, 2	                  ; reg sel output
  SBI   DDRB, 0	                  ; enable output
  SBI   PORTB, 2
  SBI   PORTB, 0
  SBI   DDRB, 1	                  ; rw output
  LDI   r16, 0x38
  RCALL Write_instruc
  LDI   r16, 0x0E                 ; turn lcd on with cursor
  RCALL Write_instruc
  LDI   r16, 0x06
  RCALL Write_instruc
  LDI   r16, 0x01
  RCALL Write_instruc
  POP   r16
  RET

 message_to_LCD:
   LDI  tmp2, 0x00                ; counter to keep track of chars on
                                  ; LCD, this works, but is brittle in
                                  ; general
 one_char:
   CPI  tmp2, 16                  ; check if we need to line break
   BRNE write_one_char
   LDI  tmp1, 0b11000000          ; instruction for next line
   CALL Write_instruc             ; see data sheet
 write_one_char:                  ; write a char from the address in X
   LD   tmp1, X+
   CPI  tmp1, 0x00
   BREQ m2ldone                   ; end on null
   CALL write_char
   INC  tmp2
   RJMP one_char                  ; do next chars
 m2ldone:
  RET

clear_LCD:
  CBI   PORTB, 0
  CBI   PORTB, 1    
  LDI   R16, 0x01                 ; write instruction to clear display
  RCALL write_instruc
  RET
