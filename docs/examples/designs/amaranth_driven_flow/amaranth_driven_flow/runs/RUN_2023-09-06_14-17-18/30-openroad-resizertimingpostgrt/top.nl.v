module top (detector_in,
    phase_map_out);
 input [1:0] detector_in;
 output [1:0] phase_map_out;

 wire _0_;
 wire _1_;
 wire net1;
 wire net2;
 wire net3;
 wire net4;

 sky130_fd_sc_hd__clkbuf_1 _2_ (.A(net2),
    .X(_0_));
 sky130_fd_sc_hd__clkbuf_1 _3_ (.A(_0_),
    .X(net3));
 sky130_fd_sc_hd__or2_1 _4_ (.A(net1),
    .B(net2),
    .X(_1_));
 sky130_fd_sc_hd__clkbuf_1 _5_ (.A(_1_),
    .X(net4));
 sky130_fd_sc_hd__decap_3 PHY_0 ();
 sky130_fd_sc_hd__decap_3 PHY_1 ();
 sky130_fd_sc_hd__decap_3 PHY_2 ();
 sky130_fd_sc_hd__decap_3 PHY_3 ();
 sky130_fd_sc_hd__decap_3 PHY_4 ();
 sky130_fd_sc_hd__decap_3 PHY_5 ();
 sky130_fd_sc_hd__decap_3 PHY_6 ();
 sky130_fd_sc_hd__decap_3 PHY_7 ();
 sky130_fd_sc_hd__decap_3 PHY_8 ();
 sky130_fd_sc_hd__decap_3 PHY_9 ();
 sky130_fd_sc_hd__decap_3 PHY_10 ();
 sky130_fd_sc_hd__decap_3 PHY_11 ();
 sky130_fd_sc_hd__decap_3 PHY_12 ();
 sky130_fd_sc_hd__decap_3 PHY_13 ();
 sky130_fd_sc_hd__decap_3 PHY_14 ();
 sky130_fd_sc_hd__decap_3 PHY_15 ();
 sky130_fd_sc_hd__decap_3 PHY_16 ();
 sky130_fd_sc_hd__decap_3 PHY_17 ();
 sky130_fd_sc_hd__decap_3 PHY_18 ();
 sky130_fd_sc_hd__decap_3 PHY_19 ();
 sky130_fd_sc_hd__decap_3 PHY_20 ();
 sky130_fd_sc_hd__decap_3 PHY_21 ();
 sky130_fd_sc_hd__decap_3 PHY_22 ();
 sky130_fd_sc_hd__decap_3 PHY_23 ();
 sky130_fd_sc_hd__decap_3 PHY_24 ();
 sky130_fd_sc_hd__decap_3 PHY_25 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_26 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_27 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_28 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_29 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_30 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_31 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_32 ();
 sky130_fd_sc_hd__clkbuf_1 input1 (.A(detector_in[0]),
    .X(net1));
 sky130_fd_sc_hd__buf_1 input2 (.A(detector_in[1]),
    .X(net2));
 sky130_fd_sc_hd__clkbuf_4 output3 (.A(net3),
    .X(phase_map_out[0]));
 sky130_fd_sc_hd__buf_2 output4 (.A(net4),
    .X(phase_map_out[1]));
endmodule
