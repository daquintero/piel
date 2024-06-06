/* Generated by Amaranth Yosys 0.40 (PyPI ver 0.40.0.0.post95, git sha1 a1bb0255d) */

(* top =  1  *)
(* generator = "Amaranth" *)
module top(phase_map_out, detector_in);
  reg \$auto$verilog_backend.cc:2352:dump_module$1  = 0;
  (* src = "/home/daquintero/phd/piel_private/piel/tools/amaranth/construct.py:47" *)
  input [1:0] detector_in;
  wire [1:0] detector_in;
  (* src = "/home/daquintero/phd/piel_private/piel/tools/amaranth/construct.py:47" *)
  output [1:0] phase_map_out;
  reg [1:0] phase_map_out;
  always @* begin
    if (\$auto$verilog_backend.cc:2352:dump_module$1 ) begin end
    (* full_case = 32'd1 *)
    (* src = "/home/daquintero/phd/piel_private/piel/tools/amaranth/construct.py:61" *)
    casez (detector_in)
      /* src = "/home/daquintero/phd/piel_private/piel/tools/amaranth/construct.py:64" */
      2'h0:
          phase_map_out = 2'h0;
      /* src = "/home/daquintero/phd/piel_private/piel/tools/amaranth/construct.py:64" */
      2'h1:
          phase_map_out = 2'h2;
      /* src = "/home/daquintero/phd/piel_private/piel/tools/amaranth/construct.py:64" */
      2'h2:
          phase_map_out = 2'h3;
      /* src = "/home/daquintero/phd/piel_private/piel/tools/amaranth/construct.py:64" */
      2'h3:
          phase_map_out = 2'h3;
    endcase
  end
endmodule
