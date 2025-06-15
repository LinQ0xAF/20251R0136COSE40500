/*
 * GccApplication.c
 *
 * Created: 2025-05-29 ?˜¤?›„ 7:23:43
 * Author : sbr07
 */ 


#pragma GCC target ("thumb")

#include "sam.h"
#include "uart_print.h"

void GCLK_setup();
void USART_setup();
void PORT_setup();
void EIC_setup();

volatile unsigned int Distance = 0;

unsigned int v_low = 300;
unsigned int v_mid = 600;
unsigned int v_high = 900;

void TC3_setup();
void TC4_setup();
void TC5_setup();

int main()
{
	/* Initialize the SAM system */
    SystemInit();
	GCLK_setup();
	USART_setup();
	PORT_setup();
	EIC_setup();
	RTC_setup();
	
	TC3_setup();
	TC4_setup();
	TC5_setup();

//
// NVIC setup for EIC (ID #4)
//

	NVIC->ISER[0] = 1 << 4 ;  // Interrupt Set Enable for EIC
	NVIC->IP[1] = 0x40 << 0 ; // priority for EIC: IP1[7:0] = 0x40 (=b0100_0000, 2-bit MSBs)
	

	while (1) {	
		PORT->Group[0].OUTSET.reg = 0x1 <<6;
		PORT->Group[0].OUTCLR.reg = 0x1 <<7;
	};	
	return (0);
}


void GCLK_setup() {
	
	// OSC8M
	SYSCTRL->OSC8M.bit.PRESC = 0;  // prescalar to 1
	SYSCTRL->OSC8M.bit.ONDEMAND = 0;

	GCLK->GENCTRL.bit.ID = 0; // Generator #0
	GCLK->GENCTRL.bit.SRC = 6; // OSC8M
	GCLK->GENCTRL.bit.OE = 1 ;  // Output Enable: GCLK_IO
	GCLK->GENCTRL.bit.GENEN = 1; // Generator Enable
	
	GCLK->CLKCTRL.bit.ID = 4; // ID #4 (RTC)
	GCLK->CLKCTRL.bit.GEN = 0; // Generator #0 selected for RTC
	GCLK->CLKCTRL.bit.CLKEN = 1; // Now, clock is supplied to RTC!	
	
	GCLK->CLKCTRL.bit.ID = 5; // ID #5 (EIC)
	GCLK->CLKCTRL.bit.GEN = 0; // Generator #0 selected for RTC
	GCLK->CLKCTRL.bit.CLKEN = 1; // Now, clock is supplied to RTC!	
	
	GCLK->CLKCTRL.bit.ID = 0x1B; // ID #ID (TCC2, TC3)
	GCLK->CLKCTRL.bit.GEN = 0; // Generator #0 selected for TCC2, TC3
	GCLK->CLKCTRL.bit.CLKEN = 1; // Now, clock is supplied to TCC2, TC3	
	
	GCLK->CLKCTRL.bit.ID = 0x1C; // ID #ID (TC4, TC5)
	GCLK->CLKCTRL.bit.GEN = 0; // Generator #0 selected for TC4, TC5
	GCLK->CLKCTRL.bit.CLKEN = 1; // Now, clock is supplied to TC4, TC5	

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

	SERCOM5->USART.BAUD.reg = 0Xc504 ; // 115,200 bps (baud rate) with 8MHz input clock

	SERCOM5->USART.CTRLB.bit.RXEN = 1 ;
	SERCOM5->USART.CTRLB.bit.TXEN = 1 ;

	SERCOM5->USART.CTRLA.bit.ENABLE = 1;
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
	PORT->Group[0].PINCFG[17].reg = 0x0;		// peripheral mux enable = 0
	PORT->Group[0].DIRSET.reg = 0x1 << 17;			// Direction: Output
	PORT->Group[0].OUT.reg = 0 << 17 ;          // Set the Trigger to 0
	
	//
	// PORT setup for PA16 to take the echo input from Ultrasonic sensor
	//
	PORT->Group[0].PINCFG[16].reg = 0x41;		// peripheral mux: DRVSTR=1, PMUXEN = 1
	PORT->Group[0].PMUX[8].bit.PMUXE = 0x0;		// peripheral function A (EIC) selected: EXTINT[3]
	
	// PORT setup for PA6,7 for motor control
	PORT->Group[0].PINCFG[6].reg = 0x0; // peripheral mux enable = 0
	PORT->Group[0].PINCFG[7].reg = 0x0; // peripheral mux enable = 0
	PORT->Group[0].DIRSET.reg = 0x3 << 6; // Direction: Output
	
	
}

void EIC_setup() {
	// Interrupt configuration for EXTINT[3] via PA16
	
	EIC->CONFIG[0].bit.FILTEN3 = 1 ;    // filter is enabled
	EIC->CONFIG[0].bit.SENSE3 = 0x3 ;   // Both-edges detection
	EIC->INTENSET.bit.EXTINT3 = 1 ;     // External Interrupt 3 is enabled
	EIC->CTRL.bit.ENABLE = 1 ;          // EIC is enabled	
	while (EIC->STATUS.bit.SYNCBUSY); 
}

void RTC_setup() {
	//
	// RTC setup: MODE0 (32-bit counter) with COMPARE 0
	//
	PM->APBAMASK.bit.RTC_ = 1;
	
	RTC->MODE0.CTRL.bit.ENABLE = 0; // Disable first
	RTC->MODE0.CTRL.bit.MODE = 0; // Mode 0
	RTC->MODE0.CTRL.bit.MATCHCLR = 1; // match clear
	
	// 8MHz RTC clock  --> 10 usec when 80 is counted
	RTC->MODE0.COMP->reg = 80; // compare register to set up 10usec interval 
	RTC->MODE0.COUNT.reg = 0x0; // initialize the counter to 0
	RTC->MODE0.CTRL.bit.ENABLE = 1; // Enable
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
	PM->APBCMASK.bit.TC4_ = 1 ; // Clock Enable (APBC clock) for TC4

	//
	// TC4 setup: 16-bit Mode
	//

	TC4->COUNT16.CTRLA.bit.MODE = 0;  // Count16 mode
	TC4->COUNT16.CTRLA.bit.WAVEGEN = 3 ; // Match PWM (MPWM)
	TC4->COUNT16.CTRLA.bit.PRESCALER = 6; // Timer Counter clock 31,250 Hz = 8MHz / 256
	TC4->COUNT16.CC[0].reg = 1000;  // CC0 defines the period
	TC4->COUNT16.CC[1].reg = 200;  // CC1 match pulls down WO[1]
	TC4->COUNT16.CTRLA.bit.ENABLE = 1;
}

void TC5_setup(){
	TC5->COUNT16.CTRLA.bit.ENABLE = 0;
	// Power Manager
	PM->APBCMASK.bit.TC5_ = 1 ; // Clock Enable (APBC clock) for TC5
	
	TC5->COUNT16.CTRLA.bit.MODE = 0;  // Count16 mode
	TC5->COUNT16.CTRLA.bit.WAVEGEN = 0x1 ; // MFRQ
	TC5->COUNT16.CTRLA.bit.PRESCALER = 0x7; // 8MHz / 1024 = 7812Hz
	TC5->COUNT16.COUNT.reg = 0;
	TC5->COUNT16.CC[0].reg = 7812;  // 1ÃÊ
	
	TC5->COUNT16.INTENSET.bit.MC0 = 1;
	NVIC->ISER[0] = (1 << 20);        // Enable TC5 interrupt in NVIC
	
	TC5->COUNT16.CTRLA.bit.ENABLE = 1;
	while (TC5->COUNT16.STATUS.bit.SYNCBUSY);
}

//
// EIC Interrupt Handler
//

unsigned int RTC_count, count_start, count_end;
unsigned int Num_EIC_interrupts = 0;

void EIC_Handler(void)
{
	unsigned int echo_time_interval, distance;
	
	EIC->INTFLAG.bit.EXTINT3 = 1 ; // Clear the EXTINT3 interrupt flag
	Num_EIC_interrupts++;
	
	if		(Num_EIC_interrupts == 1) {
		count_start = RTC->MODE0.COUNT.reg;
	}
	else if (Num_EIC_interrupts == 2) {
		
		count_end   = RTC->MODE0.COUNT.reg;
		RTC_count = count_end - count_start;
		echo_time_interval = RTC_count / 8 ; // echo interval in usec (8MHz clock input)
		distance = (echo_time_interval / 2) * 0.034 ; // distance in cm
		
		Distance = distance;	
		
		print_decimal(distance / 100);
		distance = distance % 100;
		print_decimal(distance / 10);
		print_decimal(distance % 10);
		print_enter();		
		Num_EIC_interrupts = 0 ;
		
	}
}


void TC5_Handler(void) {

	TC5->COUNT16.INTFLAG.bit.MC0 = 1; // Clear flag

	// 
	PORT->Group[0].OUTSET.reg = (1 << 17);
	for (int i = 0; i < 10; i++);                // 
	PORT->Group[0].OUTCLR.reg = (1 << 17);   // Trigger LOW
		
	unsigned int distance = Distance;
		
	if(distance <= 10) {
		TC3->COUNT16.CC[1].reg = 0;  // stop by generating no pulse
		TC4->COUNT16.CC[1].reg = 0;  // stop by generating no pulse
	}
	else if(distance <= 20) {
		TC3->COUNT16.CC[1].reg = v_low;  //
		TC4->COUNT16.CC[1].reg = v_low;  //
	}
	else if(distance <= 30) {
		TC3->COUNT16.CC[1].reg = v_mid;  //
		TC4->COUNT16.CC[1].reg = v_mid;  //
	}
	else {
		TC3->COUNT16.CC[1].reg = v_high;  //
		TC4->COUNT16.CC[1].reg = v_high;  //
	}
		
}


