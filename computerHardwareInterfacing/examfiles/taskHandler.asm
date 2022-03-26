; found this at 
; https://www.avrfreaks.net/forum/how-do-you-make-jump-table-avr-assembly
.MACRO jumpto       
  LDI   XH, HIGH(@0)  ; NB Can't use Z register for IJMP
  LDI   XL, LOW(@0)
  ADD   XL, @1
  ADC   XH, ZERO
  PUSH  XL
  PUSH  XH
  RET
.ENDMACRO

do_task:
  jumpto taskTable, tasknum
taskTable:
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
  CALL  task6
  RET
  CALL  task7
  RET
  CALL  task8
  RET
  CALL  task9
	RET
  CALL  task10
	RET
  CALL  task11
  RET

stepper_done:
  SBI   UCSRB, RXEN               ; Re-enable receiving
  LDI   tasknum, 0
  RCALL do_task
  RET

task0:
  CALL  send_menu
  RET

task1:
  CBI   UCSRB, RXEN 
  CALL  clear_terminal
  LDI   tmp1, 1
  MOV   dstep, tmp1
  LDI   arg1, 10                  ; set number of steps
  LDI   arg2, 100
  CALL  step_motor
  RET
task2:
  CBI   UCSRB, RXEN 
  CALL  clear_terminal
  LDI   tmp1, -1
  MOV   dstep, tmp1
  LDI   arg1, 10                  ; set number of steps
  LDI   arg2, 100
  CALL  step_motor
  RET
task3:
  CBI   UCSRB, RXEN 
  CALL  clear_terminal
  LDI   tmp1, 2
  MOV   dstep, tmp1
  LDI   arg1, 80                  ; set number of steps
  LDI   arg2, 25
  CALL  step_motor
  RET
task4:
  CBI   UCSRB, RXEN 
  CALL  clear_terminal
  LDI   tmp1, -2
  MOV   dstep, tmp1
  LDI   arg1, 80                  ; set number of steps
  LDI   arg2, 25
  CALL  step_motor
  RET
task5:
  CALL  stepper_disable
  LDI   tasknum, 0x00
  RCALL stepper_done
  RET
task6:
  CALL  stepper_enable
  LDI   tasknum, 0x00
  RCALL stepper_done
  RET
task7:
  RET
task8:
  RET
task9:
  RET
task10:
  RET
task11:
  RET

menu_text:          .db 0x0c,"Project Tasks: ",0x0d,0x0a
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
blankterminal:      .db 0x0c,0x00
