/*
 * GccApplication_Assign6.c
 *
 * Created: 2025-05-22 ?˜¤? „ 3:00:20
 * Author : sbr07
 */ 

#pragma GCC target ("thumb")

#include "sam.h"
#include "uart_print.h"

void GCLK_setup();
void USART_setup();
void PORT_setup();
void RTC_setup();
unsigned int Distance();

void TC3_setup();
void TC4_setup();

int main()
{
    int i;
	/* Initialize the SAM system */
    SystemInit();

	GCLK_setup();
	
	USART_setup();

	PORT_setup();
	
	TC3_setup();

	TC4_setup();
	
	unsigned int v_low = 300;
	unsigned int v_mid = 600;
	unsigned int v_high = 900;
	unsigned int distance = 0;
	
	while (1) {
		// ----------------------------------------------
		// 10 usec pulse to Trigger input in Ultrasonic sensor
		// --------------------------------------------
		PORT->Group[0].OUT.reg = 0 << 17 ;
	
		RTC_setup();
	
		while (RTC->MODE0.INTFLAG.bit.CMP0 != 1) ;

		RTC->MODE0.INTFLAG.bit.CMP0 = 1; // clear overflow interrupt flag
		PORT->Group[0].OUT.reg = 1 << 17;	

		while (RTC->MODE0.INTFLAG.bit.CMP0 != 1) ;

		RTC->MODE0.INTFLAG.bit.CMP0 = 1; // clear overflow interrupt flag
		PORT->Group[0].OUT.reg = 0 << 17 ;
		// --------------------------------------------

		// Now use RTC to measure the pulse width of echo input.
		RTC->MODE0.CTRL.bit.ENABLE = 0; // Disable first
		RTC->MODE0.CTRL.bit.MATCHCLR = 0; // Now just count...
		RTC->MODE0.COUNT.reg = 0x0; // initialize the counter to 0
		RTC->MODE0.CTRL.bit.ENABLE = 1; // Enable	
	
		// Measure the distance	
		distance = Distance();
		
		if(distance <= 10) {
			TC3->COUNT16.CC[1].reg = 0;  // stop by generating no pulse
			TC4->COUNT16.CC[1].reg = 0;  // stop by generating no pulse
		}
		else if(distance <= 20) {
			TC3->COUNT16.CC[1].reg = v_low;  //
			TC4->COUNT16.CC[1].reg = v_low;  //
			PORT->Group[0].OUT.reg = 0x1 <<6;			
		}
		else if(distance <= 30) {
			TC3->COUNT16.CC[1].reg = v_mid;  //
			TC4->COUNT16.CC[1].reg = v_mid;  //
			PORT->Group[0].OUT.reg = 0x1 <<6;
		}
		else {
			TC3->COUNT16.CC[1].reg = v_high;  //
			TC4->COUNT16.CC[1].reg = v_high;  //
			PORT->Group[0].OUT.reg = 0x1 <<6;
		}
		
		for (i= 0; i<1000000; i++ ) ;  // idle roughly for 1 second
	
	};
	

	return (0);
}


void GCLK_setup() {
	
	// OSC8M
	//SYSCTRL->OSC8M.bit.ENABLE = 0; // Disable
	SYSCTRL->OSC8M.bit.PRESC = 0;  // prescaler to 1
	SYSCTRL->OSC8M.bit.ONDEMAND = 0;
	//SYSCTRL->OSC8M.bit.ENABLE = 1; // Enable

	//
	// Generic Clock Controller setup for RTC
	// * RTC ID: #4 
	// * Generator #0 is feeding RTC
	// * Generator #0 is taking the clock source #6 (OSC8M: 8MHz clock input) as an input

	GCLK->GENCTRL.bit.ID = 0; // Generator #0
	GCLK->GENCTRL.bit.SRC = 6; // OSC8M
	GCLK->GENCTRL.bit.OE = 1 ;  // Output Enable: GCLK_IO
	GCLK->GENCTRL.bit.GENEN = 1; // Generator Enable
	
	GCLK->CLKCTRL.bit.ID = 4; // ID #4 (RTC)
	GCLK->CLKCTRL.bit.GEN = 0; // Generator #0 selected for RTC
	GCLK->CLKCTRL.bit.CLKEN = 1; // Now, clock is supplied to RTC!	
	
	
	GCLK->CLKCTRL.bit.ID = 0x1B; // ID #ID (TCC2, TC3)
	GCLK->CLKCTRL.bit.GEN = 0; // Generator #0 selected for TCC2, TC3
	GCLK->CLKCTRL.bit.CLKEN = 1; // Now, clock is supplied to TCC2, TC3
	
	GCLK->CLKCTRL.bit.ID = 0x1C; // ID #ID (TC4, TC5)
	GCLK->CLKCTRL.bit.GEN = 0; // Generator #0 selected for TC4, TC5
	GCLK->CLKCTRL.bit.CLKEN = 1; // Now, clock is supplied to TC4, TC5	
}

void PORT_setup() {
	
	//
	// PORT setup for PA14 (GCLK_IO[0]) to check out clock output using logic analyzer
	//
	//PORT->Group[0].PINCFG[14].reg = 0x41;		// peripheral mux: DRVSTR=1, PMUXEN = 1
	//PORT->Group[0].PMUX[7].bit.PMUXE = 0x07;	// peripheral function H selected

	//
	// PORT setup for PA17: Built-in LED output & Trigger in Ultrasonic Sensor
	//
	PORT->Group[0].PINCFG[17].reg = 0x0;		// PMUXEN = 0
	PORT->Group[0].DIRSET.reg = 0x1 << 17;		// Direction: Output
	PORT->Group[0].OUT.reg = 0 << 17 ;          // Set the Trigger to 0

	//
	// PORT setup for PA16 to take the echo input from Ultrasonic sensor
	//
	PORT->Group[0].PINCFG[16].reg = 0x2;		// INEN = 1, PMUXEN = 0
	PORT->Group[0].DIRCLR.reg = 0x1 << 16;		// Direction: Input
	
	// PORT setup for PA6,7 for motor control
	PORT->Group[0].PINCFG[6].reg = 0x0; // peripheral mux enable = 0
	PORT->Group[0].PINCFG[7].reg = 0x0; // peripheral mux enable = 0
	PORT->Group[0].DIRSET.reg = 0x3 << 6; // Direction: Output	

}


void RTC_setup() {
	//
	// RTC setup: MODE0 (32-bit counter) with COMPARE 0
	//
	RTC->MODE0.CTRL.bit.ENABLE = 0; // Disable first
	RTC->MODE0.CTRL.bit.MODE = 0; // Mode 0
	RTC->MODE0.CTRL.bit.MATCHCLR = 1; // match clear
	
	// 8MHz RTC clock  --> 10 usec when 80 is counted
	RTC->MODE0.COMP->reg = 80; // compare register to set up 10usec interval 
	RTC->MODE0.COUNT.reg = 0x0; // initialize the counter to 0
	RTC->MODE0.CTRL.bit.ENABLE = 1; // Enable
}


unsigned int Distance(void)
{
	unsigned int RTC_count, count_start, count_end;
	unsigned int echo_time_interval, distance;
	
	// take the echo input from PA16
	while (!(PORT->Group[0].IN.reg & 0x00010000)) ;  
		
	count_start = RTC->MODE0.COUNT.reg;

	// take the echo input from PA16
	while (PORT->Group[0].IN.reg & 0x00010000);
		
	count_end   = RTC->MODE0.COUNT.reg;
	RTC_count = count_end - count_start;
	echo_time_interval = RTC_count / 8 ; // echo interval in usec (8MHz clock input)
	distance = (echo_time_interval / 2) * 0.034 ; // distance in cm
	
	
	print_decimal(distance / 100);
	distance = distance % 100;
	print_decimal(distance / 10);
	print_decimal(distance % 10);
	print_enter();		
	
	return distance;
}


void USART_setup() {
	
	//
	// PORT setup for PB22 and PB23 (USART)
	//
	PORT->Group[1].PINCFG[22].reg = 0x41; // peripheral mux: DRVSTR=1, PMUXEN = 1
	PORT->Group[1].PINCFG[23].reg = 0x41; // peripheral mux: DRVSTR=1, PMUXEN = 1

	PORT->Group[1].PMUX[11].bit.PMUXE = 0x03; // peripheral function D selected
	PORT->Group[1].PMUX[11].bit.PMUXO = 0x03; // peripheral function D selected

	// Power Manager
	PM->APBCMASK.bit.SERCOM5_ = 1 ; // Clock Enable (APBC clock) for USART
	
	//
	// * SERCOM5: USART
	// * Generator #0 is feeding USART as well
	//
	GCLK->CLKCTRL.bit.ID = 0x19; // ID #0x19 (SERCOM5: USART): GCLK_SERCOM3_CORE
	GCLK->CLKCTRL.bit.GEN = 0; // Generator #0 selected for USART
	GCLK->CLKCTRL.bit.CLKEN = 1; // Now, clock is supplied to USART!

	GCLK->CLKCTRL.bit.ID = 0x13; // ID #0x13 (SERCOM5: USART): GCLK_SERCOM_SLOW
	GCLK->CLKCTRL.bit.GEN = 0; // Generator #0 selected for USART
	GCLK->CLKCTRL.bit.CLKEN = 1; // Now, clock is supplied to USART!
	
	//
	// USART setup
	//
	SERCOM5->USART.CTRLA.bit.MODE = 1 ; // Internal Clock
	SERCOM5->USART.CTRLA.bit.CMODE = 0 ; // Asynchronous UART
	SERCOM5->USART.CTRLA.bit.RXPO = 3 ; // PAD3
	SERCOM5->USART.CTRLA.bit.TXPO = 1 ; // PAD2
	SERCOM5->USART.CTRLB.bit.CHSIZE = 0 ; // 8-bit data
	SERCOM5->USART.CTRLA.bit.DORD = 1 ; // LSB first
	//SERCOM5->USART.CTRLA.bit.SAMPR = 1 ; //

	SERCOM5->USART.BAUD.reg = 0Xc504 ; // 115,200 bps (baud rate) with 8MHz input clock
	//SERCOM5->USART.BAUD.reg = 0Xe282 ; // 115,200 bps (baud rate) with 16MHz input clock
	//SERCOM5->USART.BAUD.reg = 0Xec57 ; // 115,200 bps (baud rate) with 24MHz input clock
	//SERCOM5->USART.BAUD.reg = 0Xf62b ; // 115,200 bps (baud rate) with 48MHz input clock

	SERCOM5->USART.CTRLB.bit.RXEN = 1 ;
	SERCOM5->USART.CTRLB.bit.TXEN = 1 ;

	SERCOM5->USART.CTRLA.bit.ENABLE = 1;
}

void TC3_setup() {

	//
	// PORT setup for PA18 ( TC3's WO[0] )
	//
	PORT->Group[0].PINCFG[18].reg = 0x41;		// peripheral mux: DRVSTR=1, PMUXEN = 1
	PORT->Group[0].PMUX[9].bit.PMUXE = 0x04;	// peripheral function E selected

	//
	// PORT setup for PA19 ( TC3's WO[1] )
	//
	PORT->Group[0].PINCFG[19].reg = 0x41;		// peripheral mux: DRVSTR=1, PMUXEN = 1
	PORT->Group[0].PMUX[9].bit.PMUXO = 0x04;	// peripheral function E selected

	// Power Manager
	PM->APBCMASK.bit.TC3_ = 1 ; // Clock Enable (APBC clock) for TC3

	//
	// TC3 setup: 16-bit Mode
	//

	TC3->COUNT16.CTRLA.bit.MODE = 0;  // Count16 mode
	TC3->COUNT16.CTRLA.bit.WAVEGEN = 3 ; // Match PWM (MPWM)
	TC3->COUNT16.CTRLA.bit.PRESCALER = 6; // Timer Counter clock 31,250 Hz = 8MHz / 256
	TC3->COUNT16.CC[0].reg = 1000;  // CC0 defines the period
	TC3->COUNT16.CC[1].reg = 200;  // CC1 match pulls down WO[1]
	TC3->COUNT16.CTRLA.bit.ENABLE = 1 ;
}


void TC4_setup() {

	//
	// PORT setup for PA22 ( TC4's WO[0] )
	//
	PORT->Group[0].PINCFG[22].reg = 0x41;		// peripheral mux: DRVSTR=1, PMUXEN = 1
	PORT->Group[0].PMUX[11].bit.PMUXE = 0x04;	// peripheral function E selected

	//
	// PORT setup for PA23 ( TC4's WO[1] )
	//
	PORT->Group[0].PINCFG[23].reg = 0x41;		// peripheral mux: DRVSTR=1, PMUXEN = 1
	PORT->Group[0].PMUX[11].bit.PMUXO = 0x04;	// peripheral function E selected

	// Power Manager
	PM->APBCMASK.bit.TC4_ = 1 ; // Clock Enable (APBC clock) for TC3

	//
	// TC4 setup: 16-bit Mode
	//

	TC4->COUNT16.CTRLA.bit.MODE = 0;  // Count16 mode
	TC4->COUNT16.CTRLA.bit.WAVEGEN = 3 ; // Match PWM (MPWM)
	TC4->COUNT16.CTRLA.bit.PRESCALER = 6; // Timer Counter clock 31,250 Hz = 8MHz / 256
	TC4->COUNT16.CC[0].reg = 1000;  // CC0 defines the period
	TC4->COUNT16.CC[1].reg = 200;  // CC1 match pulls down WO[1]
	TC4->COUNT16.CTRLA.bit.ENABLE = 1 ;
}