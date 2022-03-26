.ESEG     
EEPMESSAGE1: .DB "Message1_Hi, Bye!" 0x00     
EEPMESSAGE2: .DB "Message2_lol" 0x00     
EEPMESSAGE3: .DB "Message3_abcdefghi" 0x00     
    
.DSEG    
RAMMESSAGE1: .BYTE 20    
RAMMESSAGE2: .BYTE 20    
RAMMESSAGE3: .BYTE 20 

.CSEG
; Reads the EEPROM messages into RAM
init_EEPmessages:
  ; read first message into RAM
  LDI   YH, HIGH(EEPMESSAGE1)
  LDI   YL, LOW(EEPMESSAGE1)
  LDI   XH, HIGH(RAMMESSAGE1)
  LDI   XL, LOW(RAMMESSAGE1)
  RCALL read_EEP
  ; read second message into RAM
  LDI   YH, HIGH(EEPMESSAGE2)
  LDI   YL, LOW(EEPMESSAGE2)
  LDI   XH, HIGH(RAMMESSAGE2)
  LDI   XL, LOW(RAMMESSAGE2)
  RCALL read_EEP
  ; read third message into RAM
  LDI   YH, HIGH(EEPMESSAGE3)
  LDI   YL, LOW(EEPMESSAGE3)
  LDI   XH, HIGH(RAMMESSAGE3)
  LDI   XL, LOW(RAMMESSAGE3)
  RCALL read_EEP
  RET

; Reads an individual message into RAM, stops on null byte
; Y -- Location of first byte in EEPROM to read
; X -- Location to store byte in RAM
read_EEP:
  OUT   EEARH, YH
  OUT   EEARL, YL
  SBI   EECR, EERE  ; read from EEPROM
  IN    tmp1, EEDR
  ST    X+, tmp1
  ADIW  YL, 1
  CPI   tmp1, 0x00
  BRNE  read_EEP
  RET
