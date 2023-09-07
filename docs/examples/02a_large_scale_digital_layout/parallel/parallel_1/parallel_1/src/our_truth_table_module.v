/* Generated by Yosys 0.30+48 (git sha1 14d50a176d5, clang++ 11.1.0 -fPIC -Os) */

(* \amaranth.hierarchy  = "top" *)
(* top =  1  *)
(* generator = "Amaranth" *)
module top(\output , \input );
  reg \$auto$verilog_backend.cc:2097:dump_module$1  = 0;
  (* src = "/home/daquintero/piel/piel/tools/amaranth/construct.py:47" *)
  input [3:0] \input ;
  wire [3:0] \input ;
  (* src = "/home/daquintero/piel/piel/tools/amaranth/construct.py:47" *)
  output [3:0] \output ;
  reg [3:0] \output ;
  always @* begin
    if (\$auto$verilog_backend.cc:2097:dump_module$1 ) begin end
    (* full_case = 32'd1 *)
    (* src = "/home/daquintero/piel/piel/tools/amaranth/construct.py:61" *)
    casez (\input )
      /* src = "/home/daquintero/piel/piel/tools/amaranth/construct.py:64" */
      4'h0:
          \output  = 4'h5;
      /* src = "/home/daquintero/piel/piel/tools/amaranth/construct.py:64" */
      4'h1:
          \output  = 4'hc;
      /* src = "/home/daquintero/piel/piel/tools/amaranth/construct.py:64" */
      4'h2:
          \output  = 4'h5;
      /* src = "/home/daquintero/piel/piel/tools/amaranth/construct.py:64" */
      4'h3:
          \output  = 4'h6;
      /* src = "/home/daquintero/piel/piel/tools/amaranth/construct.py:64" */
      4'h4:
          \output  = 4'h2;
      /* src = "/home/daquintero/piel/piel/tools/amaranth/construct.py:64" */
      4'h5:
          \output  = 4'hd;
      /* src = "/home/daquintero/piel/piel/tools/amaranth/construct.py:64" */
      4'h6:
          \output  = 4'h6;
      /* src = "/home/daquintero/piel/piel/tools/amaranth/construct.py:64" */
      4'h7:
          \output  = 4'h3;
      /* src = "/home/daquintero/piel/piel/tools/amaranth/construct.py:64" */
      4'h8:
          \output  = 4'h9;
      /* src = "/home/daquintero/piel/piel/tools/amaranth/construct.py:64" */
      4'h9:
          \output  = 4'he;
      /* src = "/home/daquintero/piel/piel/tools/amaranth/construct.py:64" */
      4'ha:
          \output  = 4'h4;
      /* src = "/home/daquintero/piel/piel/tools/amaranth/construct.py:64" */
      4'hb:
          \output  = 4'h8;
      /* src = "/home/daquintero/piel/piel/tools/amaranth/construct.py:64" */
      4'hc:
          \output  = 4'h1;
      /* src = "/home/daquintero/piel/piel/tools/amaranth/construct.py:64" */
      4'hd:
          \output  = 4'hb;
      /* src = "/home/daquintero/piel/piel/tools/amaranth/construct.py:64" */
      4'he:
          \output  = 4'hf;
      /* src = "/home/daquintero/piel/piel/tools/amaranth/construct.py:64" */
      4'hf:
          \output  = 4'ha;
    endcase
  end
endmodule
