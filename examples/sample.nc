; Sample NC program – simple 3-axis pocket
; Stock: 100x100x50mm,  origin at corner
; Tool: 10mm ball-end mill (radius 5mm)
; Pocket: 20x20mm centred at (50,50), depth 5mm from top (z=45)
G21          ; metric
G90          ; absolute positioning
G0 Z60       ; safe height

; --- first pass: z=45 ---
G0 X30 Y30
G0 Z47
G1 Z45 F200
G1 X70 F500
G1 Y70
G1 X30
G1 Y30
G0 Z60

; --- second pass: z=42 (deeper) ---
G0 X32 Y32
G0 Z44
G1 Z42 F200
G1 X68 F500
G1 Y68
G1 X32
G1 Y32
G0 Z60

; --- serpentine infill pass: z=42 ---
G0 X35 Y35
G1 Z42 F200
G1 X65 F500
G0 Z60
G0 X35 Y39
G1 Z42 F200
G1 X65 F500
G0 Z60
G0 X35 Y43
G1 Z42 F200
G1 X65 F500
G0 Z60
G0 X35 Y47
G1 Z42 F200
G1 X65 F500
G0 Z60
G0 X35 Y51
G1 Z42 F200
G1 X65 F500
G0 Z60
G0 X35 Y55
G1 Z42 F200
G1 X65 F500
G0 Z60
G0 X35 Y59
G1 Z42 F200
G1 X65 F500
G0 Z60
G0 X35 Y63
G1 Z42 F200
G1 X65 F500
G0 Z60

M30          ; end of program
