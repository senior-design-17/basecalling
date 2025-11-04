`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 11/03/2025 08:59:18 PM
// Design Name: 
// Module Name: rnn_lstm_cell
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


module rnn_lstm_cell #(
    parameter int M =64, 
    parameter int M2 = 128, 
    parameter int M4 = 256
    
    )(
        input wire clk, 
        input wire rst_n, 
        input logic start, 
        output logic done, 
        
        // Weights and Biases  
        input logic signed [13:0] w_all [0:M2-1][0:M4-1], 
        input logic signed [13:0] b_all [0:M4 -1], 
        
        // State IN 
        input logic signed [13:0] h_in [0:M-1],
        input logic signed [13:0] c_in [0:M-1], 
        
        // Data In 
        input logic signed [13:0] inputs [0:M2-1], 
        
        // State Out
        output logic signed [13:0] h_out[0:M-1],
        output logic signed [13:0] c_out [0:M-1]
       
    );
    
    logic signed [13:0] gates [0:M4-1]; 
    logic signed [13:0] fi [0:M-1]; 
    logic signed [13:0] fi2 [0:M-1]; 
    logic signed [13:0] fC [0:M-1]; 
    
    logic signed [13:0] ff [0:M-1]; 
    logic signed [13:0] ff2 [0:M-1];
    
    logic signed [13:0] fo [0:M-1]; 
    logic signed [13:0] fo2 [0:M-1]; 
    
    logic signed [13:0] temp1; 
    
    logic signed [13:0] temp2; 
    
    logic signed [13:0] arr [0:M-1];  
    
endmodule
