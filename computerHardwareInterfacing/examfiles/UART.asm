;               File for handling UART and related things

init_UART:
  ;set baud rate (9600,8,n,2)
 	LDI   tmp1, 51
	LDI   tmp2, 0x00
	OUT   UBRRH, tmp2
	OUT	  UBRRL, tmp1
  ; set rx and tx enable
	SBI   UCSRB, RXEN
	SBI   UCSRB, TXEN
  ; enable uart interrupts, both transmit and receive
	SBI   UCSRB, RXCIE
	SBI   UCSRB, TXCIE
	RET

send_menu:
  LDI   ZH, HIGH(2 * menu_text)
  LDI   ZL, LOW(2 * menu_text)
  LPM   tmp1, Z+
  OUT   UDR, tmp1
  RET

clear_terminal:
  LDI   ZH, HIGH(2 * blankterminal)
  LDI   ZL, LOW(2 * blankterminal)
  LPM   tmp1, Z+
  OUT   UDR, tmp1
  RET

URXC_ISR: ; on receive
  LDI   XH, HIGH(TASK_NUM_RAM)
  LDI   XL, LOW(TASK_NUM_RAM)
  IN    tmp1, UDR                 
  CPI   tmp1, '.'
  BREQ  setTask
  OUT   UDR, tmp1                 ; echo to terminal
  LD    tmp3, X                   ; get current stored number
  LDI   tmp2, 10
  MUL   tmp3, tmp2                ; multiply tmp3 by tmp2
  MOV   tmp3, MUL_LOW
  SUBI  tmp1, '0'
  LSL   tmp1
  ADD   tmp3, tmp1
  ST    X, tmp3
  RETI
setTask:
  LD    tasknum, X
  ST    X, ZERO                   ; clear the input register.
  CALL  do_task
  RETI

UTXC_ISR: ; continue transmitting
  LPM   tmp1, Z+
  CPI   tmp1, 0x00
  BREQ  donetx
  OUT   udr, tmp1
donetx:    
  RETI

sendfromram:
  LD tmp1, x+
  CPI tmp1, 0x00
  BREQ txcexit
  OUT UDR, tmp1
txcexit:
  RETI
