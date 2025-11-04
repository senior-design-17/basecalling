`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 11/03/2025 08:56:13 PM
// Design Name: 
// Module Name: sigmoid
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module sigmoid # (
    parameter int WIDTH = 14
    )(
        input wire clk, 
        input logic start, 
        input logic signed [WIDTH-1:0] din, 
        output logic done, 
        output logic signed [WIDTH-1:0] dout
    );
    
    
endmodule
