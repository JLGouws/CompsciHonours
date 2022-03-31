;                           UART.asm
; J L Gouws
;               File for handling UART and related things

init_UART:
  ;set baud rate (9600,8,n,2)
  LDI   tmp1, 51
  LDI   tmp2, 0x00
  OUT   UBRRH, tmp2               ; Baudrate
  OUT	  UBRRL, tmp1
  ; set rx and tx enable
  SBI   UCSRB, RXEN
  SBI   UCSRB, TXEN
  ; enable uart interrupts, both transmit and receive
  SBI   UCSRB, RXCIE
  SBI   UCSRB, TXCIE
  RET

send_chars_terminal:
  LPM   tmp1, Z+                  ; outputs the characters in program
                                  ; memmory to terminal
  OUT   UDR, tmp1
  RET

clear_line:                       ; clears a line in the terminal
  LDI   ZH, HIGH(2 * blankline)   ; this is one method because it gets
  LDI   ZL, LOW(2 * blankline)    ; called often
  LPM   tmp1, Z+
  OUT   UDR, tmp1                 ;
  RET

wait_transmit:
  SBIS  UCSRA, TXC                ; wait for bit data to be sent 
  RJMP  wait_transmit
  SBI   UCSRA, TXC
  RET

URXC_ISR: ; on receive
  LDI   XH, HIGH(TASK_NUM_RAM)
  LDI   XL, LOW(TASK_NUM_RAM)
  IN    tmp1, UDR                 
  CPI   tmp1, '.'                 ; check for stop character
  BREQ  set_task
  OUT   UDR, tmp1                 ; echo to terminal
  RCALL wait_transmit
  LD    tmp3, X                   ; get current stored number
  LDI   tmp2, 10
  MUL   tmp3, tmp2                ; multiply tmp3 by tmp2
  CP    MUL_HIGH, zero
  BRNE  reset_task                ; bad input this has bad interaction
                                  ; but whatever
  MOV   tmp3, MUL_LOW
  SUBI  tmp1, '0'
  BRMI  reset_task
  ADD   tmp3, tmp1
  ST    X, tmp3
  RETI
set_task:
  LD    tasknum, X
  CPI   tasknum, 0x00
  BRLT  reset_task
  CPI   tasknum, 0x0C
  BRGE  reset_task
  RJMP  do_set_task
reset_task:
  LDI   tasknum, 0x0              ; set task num to zero
do_set_task:
  ST    X, zero                   ; clear the input register.
  CALL  do_task
  RETI
  

UTXC_ISR:                         ; continue transmitting
  LPM   tmp1, Z+
  CPI   tmp1, 0x00                ; check for byte
  BREQ  donetx
  OUT   udr, tmp1
donetx:    
  RETI
