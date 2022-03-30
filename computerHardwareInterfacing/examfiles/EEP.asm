.ESEG     
EEMSG: .DB "Why am I here",0x00,"Not this message",0x00,"The message is long",0x00

.DSEG    
RAMMESSAGE1: .BYTE 20    
RAMMESSAGE2: .BYTE 20    
RAMMESSAGE3: .BYTE 20 

.CSEG
; Reads the EEPROM messages into RAM
init_EEP:
  LDI   XH, HIGH(RAMMESSAGE1)
  LDI   XL, LOW(RAMMESSAGE1)
  LDI   YH, HIGH(EEMSG)
  LDI   YL, LOW(EEMSG)
  CALL read_EEP
  LDI   XH, HIGH(RAMMESSAGE2)
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
  SBI   EECR, EERE  ; read from EEPROM
  IN    tmp1, EEDR
  ST    X+, tmp1
  ADIW  YL, 1
  CPI   tmp1, 0x00
  BRNE  read_EEP
  RET
