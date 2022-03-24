pname=main

all: comp

.PHONY: clean


comp:   
	 avra main.asm

cpr: comp
	 avrdude -p m16 -c stk500 -P /dev/ttyACM0 -e -U flash:w:main.hex

cpre: comp
	 avrdude -p m16 -c stk500 -P /dev/ttyACM0 -e -U flash:w:main.hex
	 avrdude -p m16 -c stk500 -P /dev/ttyACM0 -e -U eeprom:w:main.eep.hex

