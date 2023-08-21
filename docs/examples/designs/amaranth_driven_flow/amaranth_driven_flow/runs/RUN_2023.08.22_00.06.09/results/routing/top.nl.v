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
 sky130_fd_sc_hd__clkbuf_1 input2 (.A(detector_in[1]),
    .X(net2));
 sky130_fd_sc_hd__clkbuf_4 output3 (.A(net3),
    .X(phase_map_out[0]));
 sky130_fd_sc_hd__clkbuf_4 output4 (.A(net4),
    .X(phase_map_out[1]));
 sky130_ef_sc_hd__decap_12 FILLER_0_0_6 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_0_18 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_0_26 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_0_29 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_0_41 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_1_3 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_1_15 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_1_27 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_1_39 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_1_47 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_2_3 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_2_15 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_2_27 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_2_29 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_2_41 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_2_47 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_3_3 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_3_15 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_3_27 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_3_39 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_3_47 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_4_3 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_4_15 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_4_27 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_4_32 ();
 sky130_fd_sc_hd__decap_4 FILLER_0_4_44 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_5_3 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_5_11 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_5_16 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_5_28 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_5_40 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_6_3 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_6_9 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_6_13 ();
 sky130_fd_sc_hd__decap_3 FILLER_0_6_25 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_6_29 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_6_41 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_6_47 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_7_3 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_7_15 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_7_27 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_7_39 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_7_47 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_8_3 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_8_15 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_8_27 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_8_29 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_8_41 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_8_47 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_9_3 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_9_15 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_9_27 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_9_39 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_9_47 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_10_3 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_10_15 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_10_27 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_10_29 ();
 sky130_fd_sc_hd__decap_6 FILLER_0_10_41 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_10_47 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_11_3 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_11_15 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_11_27 ();
 sky130_fd_sc_hd__decap_3 FILLER_0_11_39 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_11_47 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_12_6 ();
 sky130_fd_sc_hd__decap_8 FILLER_0_12_18 ();
 sky130_fd_sc_hd__fill_2 FILLER_0_12_26 ();
 sky130_ef_sc_hd__decap_12 FILLER_0_12_29 ();
 sky130_fd_sc_hd__fill_1 FILLER_0_12_41 ();
endmodule
