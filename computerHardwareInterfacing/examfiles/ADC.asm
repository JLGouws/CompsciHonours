convert_voltage:
  LDI   tmp1, 0b01100000          ; AVCC selected as reference
  OUT   ADMUX, tmp1               ; ADLAR set so most significant 8 bits are in ADCH
  LDI   tmp1, 0b11001111          ; ADC enabled conversion started no auto trigger
                                  ; 128 prescaler interrupt enabled
  OUT   ADCSRA, tmp1
  RET

adc_disable:
  LDI   tmp1, 0x00                ; AVCC selected as reference
  OUT   ADMUX, tmp1               ; ADLAR set so most significant 8 bits are in ADCH
  LDI   tmp1, 0x00                ; ADC enabled conversion started no auto trigger
                                  ; 128 prescaler interrupt enabled
  OUT   ADCSRA, tmp1

ADC_ISR:
  IN    retReg, ADCH
  CALL  adc_done
  RETI
