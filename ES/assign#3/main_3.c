/*
 * GccApplication_lab_prep.c
 *
 * Created: 7/27/2023 10:26:32 AM
 * Author : suhtw
 */ 

#pragma GCC target ("thumb")

#include "sam.h"

extern int lab_asm_port();

int main()
{
	/* Initialize the SAM system */
	SystemInit();
	
	//lab_asm_port();
	
	PORT->Group[0].PINCFG[6].reg = 0x0; // peripheral mux enable = 0
	PORT->Group[0].PINCFG[7].reg = 0x0; // peripheral mux enable = 0
	PORT->Group[0].PINCFG[8].reg = 0x1; // peripheral mux enable = 0
	PORT->Group[0].PINCFG[9].reg = 0x1; // peripheral mux enable = 0
	
	PORT->Group[0].DIR.reg = 0x3 << 6; // Direction: Output - PA6&PA7
	
	while(1){
		PORT->Group[0].OUT.reg = PORT->Group[0].IN.reg >> 2;
	}
	return 0;
}
