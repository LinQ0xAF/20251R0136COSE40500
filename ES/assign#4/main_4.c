/*
 * GccApplication_Assign4.c
 *
 * Created: 2025-05-05 오후 3:15:26
 * Author : LinQ
 */ 

/*
 * GccApplication_lab_prep.c
 *
 * Created: 7/27/2023 10:26:32 AM
 * Author : suhtw
 */ 

#pragma GCC target ("thumb")

#include "sam.h"

void GCLK_setup();
void PORT_setup();
void RTC_setup();

int main()
{
	_Bool	led;
	
	/* Initialize the SAM system */
	SystemInit();

	GCLK_setup();
	
	PORT_setup();
	
	RTC_setup();

	led = 1;

	//
	// if (CMP0 interrupt status == 1) toggle LED
	// * According to the datasheet of SAMD21, INTFLAG.CMP0 bit should be cleared by writing a one to the flag
	//   But, compiler is complaining INTFLAG.CMP0 is read-only.
	//   --> Without clearing the flag, the build-in LED won't toggle
	//
	
	while (1) {
		if (RTC->MODE0.INTFLAG.bit.CMP0) {
			PORT->Group[0].OUT.reg = led << 17; // Turn on Built-in LED: Output register
			led = led ^ 1; // toggle
			RTC->MODE0.INTFLAG.bit.CMP0 = 1; // clear overflow interrupt flag
		}
	}
	
	return (1);
	
}



void GCLK_setup() {
	
	// OSC8M
	//SYSCTRL->OSC8M.bit.ENABLE = 0; // Disable
	SYSCTRL->OSC8M.bit.PRESC = 0x06;  // prescalar=64
	SYSCTRL->OSC8M.bit.ONDEMAND = 0;
	//SYSCTRL->OSC8M.bit.ENABLE = 1; // Enable

	//
	// Generic Clock Controller setup for RTC
	// * RTC ID: #4
	// * Generator #0 is feeding RTC
	// * Generator #0 is taking the clock source #6 (OSC8M: 8MHz clock input) as an input
	//
	
	GCLK->GENCTRL.bit.ID = 0; // Generator #0
	GCLK->GENCTRL.bit.SRC = 6; // OSC8M
	GCLK->GENCTRL.bit.OE = 1 ;  // Output Enable: GCLK_I	
	GCLK->GENCTRL.bit.GENEN = 1; // Generator Enable
	
	GCLK->CLKCTRL.bit.ID = 4; // ID #4 (RTC)
	GCLK->CLKCTRL.bit.GEN = 0; // Generator #0 selected for RTC
	GCLK->CLKCTRL.bit.CLKEN = 1; // Now, clock is supplied to RTC!

}

void PORT_setup() {

	//
	// PORT setup for PA14 (GCLK_IO[0]) to check out clock output using logic analyzer
	//
	PORT->Group[0].PINCFG[14].reg = 0x41;		// peripheral mux: DRVSTR=1, PMUXEN = 1
	PORT->Group[0].PMUX[7].bit.PMUXE = 0x07;	// peripheral function H selected
	
	//
	// PORT setup for PA17: Built-in LED 
	//
	PORT->Group[0].PINCFG[17].reg = 0x0; // peripheral mux enable = 0
	PORT->Group[0].DIR.reg = 0x1 << 17; // Direction: Output
	PORT->Group[0].OUT.reg = 0 << 17 ;

}

void RTC_setup() {

	//
	// RTC setup: MODE0 (32-bit counter) with COMPARE 0
	//

	RTC->MODE0.CTRL.bit.ENABLE = 0; // Disable first
	RTC->MODE0.CTRL.bit.MODE = 0; // Mode 0
	RTC->MODE0.CTRL.bit.MATCHCLR = 1; // match clear
	
	RTC->MODE0.COMP->reg = 125; // compare register to set up the period
	//RTC->MODE0.COMP->reg = 0x10000; // compare register	to set up the peroid
	RTC->MODE0.COUNT.reg = 0x0; // initialize the counter to 0
	RTC->MODE0.CTRL.bit.ENABLE = 1; // Enable
}