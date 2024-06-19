module top (bit_phase_0,
    bit_phase_1,
    input_fock_state_str);
 output [4:0] bit_phase_0;
 output [4:0] bit_phase_1;
 input [2:0] input_fock_state_str;

 wire net1;
 wire net2;
 wire net3;
 wire net4;
 wire net5;
 wire net6;
 wire net7;
 wire net8;
 wire net9;
 wire net10;
 wire net11;
 wire net12;
 wire net13;

 sky130_fd_sc_hd__nor3b_2 _0_ (.A(net3),
    .B(net2),
    .C_N(net1),
    .Y(net13));
 sky130_fd_sc_hd__nor3b_2 _1_ (.A(net3),
    .B(net1),
    .C_N(net2),
    .Y(net8));
 sky130_fd_sc_hd__clkbuf_1 _2_ (.A(net8),
    .X(net4));
 sky130_fd_sc_hd__clkbuf_1 _3_ (.A(net8),
    .X(net5));
 sky130_fd_sc_hd__clkbuf_1 _4_ (.A(net8),
    .X(net6));
 sky130_fd_sc_hd__clkbuf_1 _5_ (.A(net8),
    .X(net7));
 sky130_fd_sc_hd__clkbuf_1 _6_ (.A(net13),
    .X(net9));
 sky130_fd_sc_hd__clkbuf_1 _7_ (.A(net13),
    .X(net10));
 sky130_fd_sc_hd__clkbuf_1 _8_ (.A(net13),
    .X(net11));
 sky130_fd_sc_hd__clkbuf_1 _9_ (.A(net13),
    .X(net12));
 sky130_fd_sc_hd__decap_3 PHY_EDGE_ROW_0_Right_0 ();
 sky130_fd_sc_hd__decap_3 PHY_EDGE_ROW_1_Right_1 ();
 sky130_fd_sc_hd__decap_3 PHY_EDGE_ROW_2_Right_2 ();
 sky130_fd_sc_hd__decap_3 PHY_EDGE_ROW_3_Right_3 ();
 sky130_fd_sc_hd__decap_3 PHY_EDGE_ROW_4_Right_4 ();
 sky130_fd_sc_hd__decap_3 PHY_EDGE_ROW_5_Right_5 ();
 sky130_fd_sc_hd__decap_3 PHY_EDGE_ROW_6_Right_6 ();
 sky130_fd_sc_hd__decap_3 PHY_EDGE_ROW_7_Right_7 ();
 sky130_fd_sc_hd__decap_3 PHY_EDGE_ROW_8_Right_8 ();
 sky130_fd_sc_hd__decap_3 PHY_EDGE_ROW_9_Right_9 ();
 sky130_fd_sc_hd__decap_3 PHY_EDGE_ROW_0_Left_10 ();
 sky130_fd_sc_hd__decap_3 PHY_EDGE_ROW_1_Left_11 ();
 sky130_fd_sc_hd__decap_3 PHY_EDGE_ROW_2_Left_12 ();
 sky130_fd_sc_hd__decap_3 PHY_EDGE_ROW_3_Left_13 ();
 sky130_fd_sc_hd__decap_3 PHY_EDGE_ROW_4_Left_14 ();
 sky130_fd_sc_hd__decap_3 PHY_EDGE_ROW_5_Left_15 ();
 sky130_fd_sc_hd__decap_3 PHY_EDGE_ROW_6_Left_16 ();
 sky130_fd_sc_hd__decap_3 PHY_EDGE_ROW_7_Left_17 ();
 sky130_fd_sc_hd__decap_3 PHY_EDGE_ROW_8_Left_18 ();
 sky130_fd_sc_hd__decap_3 PHY_EDGE_ROW_9_Left_19 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_TAPCELL_ROW_0_20 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_TAPCELL_ROW_0_21 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_TAPCELL_ROW_1_22 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_TAPCELL_ROW_2_23 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_TAPCELL_ROW_3_24 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_TAPCELL_ROW_4_25 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_TAPCELL_ROW_5_26 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_TAPCELL_ROW_6_27 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_TAPCELL_ROW_7_28 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_TAPCELL_ROW_8_29 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_TAPCELL_ROW_9_30 ();
 sky130_fd_sc_hd__tapvpwrvgnd_1 TAP_TAPCELL_ROW_9_31 ();
 sky130_fd_sc_hd__buf_1 input1 (.A(input_fock_state_str[0]),
    .X(net1));
 sky130_fd_sc_hd__buf_1 input2 (.A(input_fock_state_str[1]),
    .X(net2));
 sky130_fd_sc_hd__buf_1 input3 (.A(input_fock_state_str[2]),
    .X(net3));
 sky130_fd_sc_hd__clkbuf_4 output4 (.A(net4),
    .X(bit_phase_0[0]));
 sky130_fd_sc_hd__buf_2 output5 (.A(net5),
    .X(bit_phase_0[1]));
 sky130_fd_sc_hd__buf_2 output6 (.A(net6),
    .X(bit_phase_0[2]));
 sky130_fd_sc_hd__buf_2 output7 (.A(net7),
    .X(bit_phase_0[3]));
 sky130_fd_sc_hd__clkbuf_4 output8 (.A(net8),
    .X(bit_phase_0[4]));
 sky130_fd_sc_hd__buf_2 output9 (.A(net9),
    .X(bit_phase_1[0]));
 sky130_fd_sc_hd__buf_2 output10 (.A(net10),
    .X(bit_phase_1[1]));
 sky130_fd_sc_hd__buf_2 output11 (.A(net11),
    .X(bit_phase_1[2]));
 sky130_fd_sc_hd__buf_2 output12 (.A(net12),
    .X(bit_phase_1[3]));
 sky130_fd_sc_hd__buf_2 output13 (.A(net13),
    .X(bit_phase_1[4]));
endmodule
