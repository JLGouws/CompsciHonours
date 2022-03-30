init_watchdog:
  LDI   tmp1, 0x0A                ; Enable watchdog 65ms time out
  OUT   WDTCR, tmp1
  RET
