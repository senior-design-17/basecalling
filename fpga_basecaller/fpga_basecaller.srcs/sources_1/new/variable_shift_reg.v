`timescale 1ns / 1ps

module variable_shift_reg #(
    parameter WIDTH = 8, 
    parameter SIZE = 3) 
(
    input clk,
    input ce,
    input rst,
    input [WIDTH-1:0] d,
    output [WIDTH-1:0] out
);
reg [WIDTH-1:0] sr [SIZE-1:0];

integer i; 

always @(posedge clk) begin 
   if (ce) begin 
        sr[0] <= d; 
        for ( i =1; i < SIZE; i = i+ 1) begin 
            sr[i] <= sr[i-1];
        end
    end
end

assign out = sr[SIZE-1];

endmodule