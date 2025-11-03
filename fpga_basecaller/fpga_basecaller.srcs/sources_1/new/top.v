`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: Xgome 
// Engineer: Nhlanhla Mavuso
// 
// Create Date: 11/03/2025 02:31:06 PM
// Design Name: 
// Module Name: top
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

module fpga_basecaller #(
        BYTE_SIZE_IN = 16,
        STREAM_SIZE = 1, 
        WEIGHT_SIZE = 128,
        CNN_LAYERS = 7, 
        BYTE_SIZE_OUT = 32,
        LSTM_LAYERS = 6 
        )(
        input wire clk,
        input logic reset_n,
        input logic [WEIGHTS_SIZE - 1 : 0] weights, 
        input logic chunk_input_header [0:STREAM_SIZE],
        output logic chunk_output_header,
        //  Reading 
        output wire rd_done,
        output wire fpga_ready,
        input wire data_ready,
        // Writing
        output wire wd_ready, // MATRIX is ready to be stored in memory 
        input wire mem_ready,  // MEMORY IS FREE or NOT FULL! can we safely write to memory,
    
        output logic [BYTE_SIZE_OUT - 1:0] output_matrix [0:1667][0:384]
    );

    logic [$clog2(FIFO_SIZE_IN)- 1: 0] fifo_in_mem [0:FIFO_SIZE_IN];
    logic [$clog2(FIFO_SIZE_OUT)- 1: 0] fifo_out_mem [0:FIFO_SIZE_OUT]; 

    logic [BYTE_SIZE_OUT - 1:0] output_matrix_internal [0:1667][0:384]; 
    
    
    // Input processing  
    
       
    // Convolutional Stack 
    
    
    // LSTM Stack 
    
    
    // CNN --->
    cnn #( .WEIGHT_SIZE(WEIGHT_SIZE)) cnn_i
        (
           .clk (clk),
           .rst_n(rst_n),
           .weights (weights)
        ); 
    // LSTM
    lstm #(.LSTM_LAYERS (LSTM_LAYERS) ) lstm_i
        (
            .clk (clk),
            .rst_n (rst_n)
        );

