module top (bit_phase_0,
    bit_phase_1,
    input_fock_state_str);
 output [4:0] bit_phase_0;
 output [4:0] bit_phase_1;
 input [2:0] input_fock_state_str;


 sky130_fd_sc_hd__nor3b_2 _0_ (.A(input_fock_state_str[2]),
    .B(input_fock_state_str[1]),
    .C_N(input_fock_state_str[0]),
    .Y(bit_phase_1[4]));
 sky130_fd_sc_hd__nor3b_2 _1_ (.A(input_fock_state_str[2]),
    .B(input_fock_state_str[0]),
    .C_N(input_fock_state_str[1]),
    .Y(bit_phase_0[4]));
 sky130_fd_sc_hd__buf_2 _2_ (.A(bit_phase_0[4]),
    .X(bit_phase_0[0]));
 sky130_fd_sc_hd__buf_2 _3_ (.A(bit_phase_0[4]),
    .X(bit_phase_0[1]));
 sky130_fd_sc_hd__buf_2 _4_ (.A(bit_phase_0[4]),
    .X(bit_phase_0[2]));
 sky130_fd_sc_hd__buf_2 _5_ (.A(bit_phase_0[4]),
    .X(bit_phase_0[3]));
 sky130_fd_sc_hd__buf_2 _6_ (.A(bit_phase_1[4]),
    .X(bit_phase_1[0]));
 sky130_fd_sc_hd__buf_2 _7_ (.A(bit_phase_1[4]),
    .X(bit_phase_1[1]));
 sky130_fd_sc_hd__buf_2 _8_ (.A(bit_phase_1[4]),
    .X(bit_phase_1[2]));
 sky130_fd_sc_hd__buf_2 _9_ (.A(bit_phase_1[4]),
    .X(bit_phase_1[3]));
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
endmodule
