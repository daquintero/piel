* `ngspice` Sim Input for `__main__.TransientTb`
* Generated by `vlsirtools.NgspiceNetlister`
* 
* Anonymous `circuit.Package`
* Generated by `vlsirtools.NgspiceNetlister`
* 

.SUBCKT TransientTb
+ VSS 
* No parameters

vVDC
+ VDC_p VSS 
+ pulse ('0m' '1000m' '1m' '1m' '1m' '10m' '10m') 
* No parameters


.ENDS

xtop 0 TransientTb // Top-Level DUT 


.tran 0.0001 0.1




