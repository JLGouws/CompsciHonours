;                         EEP.asm
; This file handles EEPROM
; J L Gouws
.ESEG     
EEMSG: .DB "Why am I here",0x00,"Not this message",0x00,"The message is long",0x00

.DSEG    
RAMMESSAGE1: .BYTE 20    
RAMMESSAGE2: .BYTE 20    
RAMMESSAGE3: .BYTE 20 

.CSEG
; Reads the EEPROM messages into RAM
init_EEP:
  LDI   XH, HIGH(RAMMESSAGE1)     ; point X to the ram message
  LDI   XL, LOW(RAMMESSAGE1)
  LDI   YH, HIGH(EEMSG)           ; set Y to point to the messages in
                                  ; EEPROM
  LDI   YL, LOW(EEMSG)
  CALL read_EEP                   ; read from EEPROM
  LDI   XH, HIGH(RAMMESSAGE2)     ; rinse an repeat
  LDI   XL, LOW(RAMMESSAGE2)
  CALL read_EEP
  LDI   XH, HIGH(RAMMESSAGE3)
  LDI   XL, LOW(RAMMESSAGE3)
  CALL read_EEP
  RET

; Reads an individual message into RAM, stops on null byte
; Y -- Location of first byte in EEPROM to read
; X -- Location to store byte in RAM
read_EEP:
  OUT   EEARH, YH
  OUT   EEARL, YL
  SBI   EECR, EERE                ; read from EEPROM
  IN    tmp1, EEDR
  ST    X+, tmp1
  ADIW  YL, 1
  CPI   tmp1, 0x00                ; see if we are at end of message
  BRNE  read_EEP                  ; read next byte
  RET
