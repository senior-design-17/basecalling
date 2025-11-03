`timescale 1ns / 1ps


module qadd #(
	parameter N = 16,
	parameter Q = 12
	)
	(
    input [N-1:0] a,
    input [N-1:0] b,
    output [N-1:0] c
    );
assign c = a + b;

endmodule