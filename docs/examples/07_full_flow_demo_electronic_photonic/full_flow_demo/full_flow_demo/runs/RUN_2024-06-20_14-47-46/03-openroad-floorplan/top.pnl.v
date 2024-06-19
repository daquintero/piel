module top (bit_phase_0,
    bit_phase_1,
    input_fock_state_str);
 output [4:0] bit_phase_0;
 output [4:0] bit_phase_1;
 input [2:0] input_fock_state_str;


 sky130_fd_sc_hd__nor3b_2 _0_ (.A(input_fock_state_str[2]),
    .B(input_fock_state_str[1]),
    .C_N(input_fock_state_str[0]),
    .VGND(VGND),
    .VNB(VGND),
    .VPB(VPWR),
    .VPWR(VPWR),
    .Y(bit_phase_1[4]));
 sky130_fd_sc_hd__nor3b_2 _1_ (.A(input_fock_state_str[2]),
    .B(input_fock_state_str[0]),
    .C_N(input_fock_state_str[1]),
    .VGND(VGND),
    .VNB(VGND),
    .VPB(VPWR),
    .VPWR(VPWR),
    .Y(bit_phase_0[4]));
 sky130_fd_sc_hd__buf_2 _2_ (.A(bit_phase_0[4]),
    .VGND(VGND),
    .VNB(VGND),
    .VPB(VPWR),
    .VPWR(VPWR),
    .X(bit_phase_0[0]));
 sky130_fd_sc_hd__buf_2 _3_ (.A(bit_phase_0[4]),
    .VGND(VGND),
    .VNB(VGND),
    .VPB(VPWR),
    .VPWR(VPWR),
    .X(bit_phase_0[1]));
 sky130_fd_sc_hd__buf_2 _4_ (.A(bit_phase_0[4]),
    .VGND(VGND),
    .VNB(VGND),
    .VPB(VPWR),
    .VPWR(VPWR),
    .X(bit_phase_0[2]));
 sky130_fd_sc_hd__buf_2 _5_ (.A(bit_phase_0[4]),
    .VGND(VGND),
    .VNB(VGND),
    .VPB(VPWR),
    .VPWR(VPWR),
    .X(bit_phase_0[3]));
 sky130_fd_sc_hd__buf_2 _6_ (.A(bit_phase_1[4]),
    .VGND(VGND),
    .VNB(VGND),
    .VPB(VPWR),
    .VPWR(VPWR),
    .X(bit_phase_1[0]));
 sky130_fd_sc_hd__buf_2 _7_ (.A(bit_phase_1[4]),
    .VGND(VGND),
    .VNB(VGND),
    .VPB(VPWR),
    .VPWR(VPWR),
    .X(bit_phase_1[1]));
 sky130_fd_sc_hd__buf_2 _8_ (.A(bit_phase_1[4]),
    .VGND(VGND),
    .VNB(VGND),
    .VPB(VPWR),
    .VPWR(VPWR),
    .X(bit_phase_1[2]));
 sky130_fd_sc_hd__buf_2 _9_ (.A(bit_phase_1[4]),
    .VGND(VGND),
    .VNB(VGND),
    .VPB(VPWR),
    .VPWR(VPWR),
    .X(bit_phase_1[3]));
endmodule
