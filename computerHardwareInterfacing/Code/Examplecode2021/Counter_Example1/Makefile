pname=main

all: comp

.PHONY: clean

comp:   
	 avra $(pname).asm

cpr: comp
   avrdude -p m16 -c stk500 -P /dev/ttyACM0 -e -E flash:w:$(main).hex
