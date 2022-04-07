;                         taskHandler.asm
; This file handles all the tasks at a high level
; J L Gouws
; found this at 
; https://www.avrfreaks.net/forum/how-do-you-make-jump-table-avr-assembly
.DSEG 
TASK_NUM_RAM: .BYTE 2 ;

.MACRO jumpto       
  LDI   XH, HIGH(@0)  ; NB Can't use Z register for IJMP
  LDI   XL, LOW(@0)
  MOV   tmp1, @1
  LSL   tmp1
  ADD   XL, tmp1
  ADC   XH, ZERO
  PUSH  XL
  PUSH  XH
  RET
.ENDMACRO

.CSEG
do_task:
  LDI   ZH, HIGH(2 * blankline)
  LDI   ZL, LOW(2 * blankline)
  CALL  send_chars_terminal
  jumpto taskTable, tasknum
taskTable:                        ; this is a jimmyied jump table
                                  ; that jumps to the correct task
  RCALL task0
  RET
  RCALL task1
  RET
  RCALL task2
  RET
  RCALL task3
  RET
  RCALL task4
  RET
  RCALL task5
  RET
  RCALL  task6
  RET
  RCALL  task7
  RET
  RCALL  task8
  RET
  RCALL  task9
	RET
  RCALL  task10
	RET
  RCALL  task11
  RET

task0:
  LDI   ZH, HIGH(2 * menu_text)   ; print out menu
  LDI   ZL, LOW(2 * menu_text)
  CALL  send_chars_terminal
  RET
task1:
  LDI   tmp1, 1
  MOV   dstep, tmp1
  LDI   arg1, 10                  ; set number of steps
  LDI   arg2, 100                 ; number of 5ms per step
  RJMP  do_motor_task
task2:
  LDI   tmp1, -1
  MOV   dstep, tmp1
  LDI   arg1, 10                  ; set number of steps
  LDI   arg2, 100                 ; number of 5ms per step
  RJMP  do_motor_task
task3:
  LDI   tmp1, 2
  MOV   dstep, tmp1
  LDI   arg1, 80                  ; set number of steps
  LDI   arg2, 25                  ; number of 5ms per step
  RJMP  do_motor_task
task4:
  LDI   tmp1, -2
  MOV   dstep, tmp1
  LDI   arg1, 80                  ; set number of steps
  LDI   arg2, 25                  ; number of 5ms per step
  RJMP  do_motor_task
task5:
  LDI   tasknum, 0x00             ; set to task 0
  CALL  stepper_disable           ; disable the motor
  RET
task6:
  LDI   tasknum, 0x00             ; set to task 0
  CALL  stepper_enable            ; disable the motor
  RET
task7:
  LDI   tasknum, 0x00
  CALL  convert_voltage           ; start adc conversion
  RET
task8:
  LDI   tasknum, 0x00
  CALL  adc_disable               ; disable the LEDs
  CALL  clear_lights              ; clear LEDs
  RET
task9:
  LDI   tasknum, 0x00
  LDI   XH, HIGH(RAMMESSAGE3)     ; point X to ram message
  LDI   XL, LOW(RAMMESSAGE3) 
  CALL  message_to_LCD            ; print message to LCD
  RET
task10:
  LDI   tasknum, 0x00
  CALL  clear_LCD                 ; clear the LCD
  RET
task11:
  NOP
  NOP
  RJMP  task11                    ; infinite loop so WD doesn't get reset
  RET

do_motor_task:
  CALL  stepper_is_enabled        ; check if the motor is actually
                                  ; enabled
  SBRS  retReg, 0
  RET
  CBI   UCSRB, RXEN 
  LDI   ZH, HIGH(2 * blankterminal)
  LDI   ZL, LOW(2 * blankterminal)
  CALL  send_chars_terminal       ; clear the terminal
  CALL  enable_buttons            ; enable the buttons
  CALL  step_motor
  RET
  

adc_done:
  MOV   arg1, retReg              ; output the lights to
  CALL  output_lights
  RET

stepper_done:
  LDI   tasknum, 0x00
  SBI   UCSRB, RXEN               ; Re-enable receiving
  CALL  disable_buttons           ; disable buttons
  RCALL do_task
  RET

menu_text:          .db 0x0C,"Project Tasks: ",0x0d,0x0a
menu_text1:         .db "--------------",0x0d,0x0a
menu_text2:         .db "1) Rotate clockwise for 5 seconds ",0x0d,0x0a
menu_text3:         .db "2) Rotate anti-clockwise for 5 seconds",0x0d,0x0a
menu_text4:         .db "3) Rotate clockwise for 10 seconds",0x0d,0x0a
menu_text5:         .db "4) Rotate anti-clockwise for 10 seconds ",0x0d,0x0a
menu_text6:         .db "5) Disable stepper",0x0d,0x0a
menu_text7:         .db "6) Enable stepper ",0x0d,0x0a
menu_text8:         .db "7) Start ADC PWM task ",0x0d,0x0a
menu_text9:         .db "8) Stop ADC PWM task",0x0d,0x0a
menu_text10:        .db "9) Print 3rd stored message to LCD",0x0d,0x0a
menu_text11:        .db "10) Clear LCD ",0x0d,0x0a
menu_text12:        .db "11) Reset Microcontroller ",0x0d,0x0a,0x00,0x00
blankterminal:      .db 0x0C,0x00
blankline:          .db 0x0D,"          ", 0x0D,0x00,0x00
