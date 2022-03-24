;Stepper Example 3
; Hardware connections.
; pd4->drv0
; pd5->drv1
; pd6->drv2
; pd7->drv3
; pd3 -> button0
; pd2 - button1

.INCLUDEPATH "/usr/share/avra" ;set the inlude path to the correct place

.NOLIST
.INCLUDE "m16def.inc"
.LIST

.def zero=R0        ; reserve R0 as our ZERO register
.def step=R1        ; will now contain the index into steptable of the current step
.def slow=R19
.def TMP1=R16		;defines serve the same purpose as in C,
.def TMP2=R17		;before assembly, defined values are substituted
.def TMP3=R18
.dseg
STEP_table: .byte 10

.cseg			;Tell the assembler that everything below this is in the code segment
.org $000		;locate code at address $000
jmp START		; Jump to the START Label
.org INT0addr
jmp int0isr
.org OVF0addr   ; locate next instruction at Timer0's overflow interrupt vector
jmp T0ovf_ISR      ; Jump to the interrupt service routine
.org OC0addr
jmp T0OC_ISR

.org $02A		;locate code past the interupt vectors
START:
    ldi TMP1, LOW(RAMEND)	;initialise the stack pointer
	out SPL, TMP1
	ldi TMP1, HIGH(RAMEND)
	out SPH, TMP1
	call read_steptable
	CALL INITIALISE_IO ; Call the subroutine INITIALISE
	call initialise_t0 ; call subroutine to init t0 with output comparevalue
    SEI ; enable global interrupts
MAIN_LOOP:
	NOP
	NOP
	RJMP MAIN_LOOP

READ_STEPTABLE:
    ;For now I am going to manually populate the steptable in SRAM
    ; single steps
    ldi xl, low(STEP_table)
    ldi xh, high(STEP_table)
    ldi tmp1, 0x00
    st x+, tmp1
    ldi tmp1, 0x10
    st x+, tmp1
    ldi tmp1, 0x30
    st x+, tmp1
    ldi tmp1, 0x20
    st x+, tmp1
    ldi tmp1, 0x60
    st x+, tmp1
    ldi tmp1, 0x40
    st x+, tmp1
    ldi tmp1, 0xC0
    st x+, tmp1
    ldi tmp1, 0x80
    st x+, tmp1
    ldi tmp1, 0x90
    st x+, tmp1
    ldi tmp1, 0x00
    st x+, tmp1
    ret

Initialise_t0:
    ldi tmp1, 0xf8 ;
    out ocr0, tmp1 ; setup oc0
    ldi tmp1, 0b00000010
    out timsk, tmp1 ; enable t0 ocint - we use the output compare intterupt in this case
                    ; because we will set the timer to CTC mode
    ldi tmp1, 0b00001101; ctc mode, no OC pin use, /1024 prescaler
    out TCCR0, tmp1
    ret

INITIALISE_IO:
    ldi tmp1, 0xF0;
    out DDRD, tmp1;
    ldi tmp1, 0x01; 1 because we want to be past the null on the boundary -
    mov step, tmp1
    sbi portd, 2 ; pullup on pd2 pin for int 0
    ldi tmp1, 0b01000000
    out GICR, tmp1 ; int0 enabled
    ldi tmp1, 0b00000010
    out mcucr, tmp1 ;falling edge triggered.
    RET					;Return from subroutine

T0ovf_ISR:

    RETI
T0OC_ISR:
inc slow
cpi slow,0x10
brne notnow
mov slow, zero
call Nextstep
notnow:
    RETI

int0isr:
;call nextstep
   RETI

NEXTSTEP:
inc step ; get to the next step
; get correct value from lookuptable
getstep:
ldi xl, low(STEP_table)
ldi xh, high(STEP_table)
add xl, step
adc xh, zero
ld tmp2, X
cpi tmp2, 0x00
brne outstep
 ; if we got a null then we need to wrap around
  ldi tmp2, 0x01 ;using tmp2 since we need the value in tmp1 and tmp2's value is rubbish at this moment'
  mov step, tmp2
  rjmp getstep
outstep:
  IN tmp1, PORTD
  andi tmp1, 0x0F ;tmp1 now contains the masked off values of portD
  or tmp1, tmp2
  out portd, tmp1
  RET

