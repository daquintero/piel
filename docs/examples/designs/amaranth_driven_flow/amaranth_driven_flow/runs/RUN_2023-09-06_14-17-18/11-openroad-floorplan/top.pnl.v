module top (detector_in,
    phase_map_out);
 input [1:0] detector_in;
 output [1:0] phase_map_out;

 wire _0_;
 wire _1_;

 sky130_fd_sc_hd__buf_1 _2_ (.A(detector_in[1]),
    .VGND(VGND),
    .VNB(VGND),
    .VPB(VPWR),
    .VPWR(VPWR),
    .X(_0_));
 sky130_fd_sc_hd__buf_1 _3_ (.A(_0_),
    .VGND(VGND),
    .VNB(VGND),
    .VPB(VPWR),
    .VPWR(VPWR),
    .X(phase_map_out[0]));
 sky130_fd_sc_hd__or2_2 _4_ (.A(detector_in[0]),
    .B(detector_in[1]),
    .VGND(VGND),
    .VNB(VGND),
    .VPB(VPWR),
    .VPWR(VPWR),
    .X(_1_));
 sky130_fd_sc_hd__buf_1 _5_ (.A(_1_),
    .VGND(VGND),
    .VNB(VGND),
    .VPB(VPWR),
    .VPWR(VPWR),
    .X(phase_map_out[1]));
endmodule
