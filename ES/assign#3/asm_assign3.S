
/*
 * asm_assign3.S
 *
 * Created: 2025-04-16 오후 9:05:14
 *  Author: LinQ
 */ 

 .syntax unified
  
 .section .text
 .globl lab_asm_port
 
 .thumb
 .thumb_func

 lab_asm_port:
	
	ldr r0, =0x41000000  // PAC1 Write Protect Clear
	ldr r1, [r0]         //
	/*
	ldr r0, =0x41000004  // PAC1 Write Protect Set
	ldr r2, [r0]         //
	*/

	ldr r0, =0x41004446  // Pin Configuration Register: PA6
	ldr r1, =0x41004447	 // Pin Configuration Register: PA7
	ldrb r2, [r0]         // check out if PA6 PINCFG is 0
	ldrb r2, [r1]		  // check out if PA7 PINCFG is 0
	ldr  r3, =0x0         // bit 1(INEN) = 0, bit 0(PMUXEN) = 0
	strb r3, [r0]         // 	 
	strb r3, [r1]		  //
	ldrb r4, [r0]		//check out if PA6 PINCFG is 0
	ldrb r4, [r1]		//check out if PA7 PINCFG is 0

	ldr r0, =0x41004448  // Pin Configuration Register: PA8
	ldr r1, =0x41004449	 // Pin Configuration Register: PA9
	ldrb r2, [r0]         // check out if PA8 PINCFG is 0
	ldrb r2, [r1]		  // check out if PA9 PINCFG is 0
	ldr  r3, =0x2         // bit 2(PULLEN) = 0, bit 1(INEN) = 1, bit 0(PMUXEN) = 0
	strb r3, [r0]         // 	 
	strb r3, [r1]
	ldrb r4, [r0]		//check out if it is 0
	ldrb r4, [r1]

	ldr r0, =0x41004400  // Direction Register
	ldr r1, =0x000000C0  // bit 6, 7 = 1 / bit 8, 9 = 0(그대로) -> 0x0000_00C0
	str r1, [r0]         // 

	ldr r0, =0x41004420		// Input Register
	ldr r1, =0x41004410		// Output Register
	ldr r6, =0x00000300		// bit 8, 9 isolate
	ldr r4, =0x0
	str r4, [r1]			// clear output once
	ldr r2, [r0]			// input register check
	ands r2, r2, r6			// 시작 전 input [8,9] 체크

wait_for_input:
	
	ldr r3, [r0]			// Check Input
	ands r3, r3, r6			// in reg의 bit 8, 9 isolate
	lsrs r3, r3, #2			// PA8, PA9의 input을 PA6, PA7로 2-bit씩 shift right
	str r3, [r1]			// shift 된 값을 저장

	b wait_for_input		// 다시 input 대기



