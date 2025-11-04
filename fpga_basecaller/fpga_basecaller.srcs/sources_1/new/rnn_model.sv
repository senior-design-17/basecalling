`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 11/03/2025 06:32:24 PM
// Design Name: 
// Module Name: rnn_model
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


module rnn_model #(
    N = 7000, 
    M = 64, 
    S = 6, 
    M2 = 128, 
    M4 = 256, 
    NP = 3, 
    MP = 64, 
    SP = 6
)(
    input wire clk, 
    input wire rst_n, 
    input logic start,
    output logic done, 
    output logic idle, 
    output logic ready, 
    input logic signed [8:0] signals [0:N-1][0:M-1], 
    input logic signed [13:0] weights_in [0:M-1][0:M-1],
    input logic signed [13:0] biases_in [0:M-1],
    input logic signed [13:0] w_hidd2 [0:M-1][0:M-1],
    input logic signed [13:0] b_hidd2  [0:M-1],
    input logic signed [13:0] w_hidd3 [0:M-1][0:M-1],
    input logic signed [13:0] b_hidd3  [0:M-1],
    input logic signed [13:0] w_all [0:M2-1][0:M4-1],
    input logic signed [13:0] b_all [0:M4-1],
    input logic signed [13:0] w_all2 [0:M2-1][0:M4-1],
    input logic signed [13:0] b_all2 [0:M4-1],
    input logic signed [13:0] w_out [0:M-1][0:S-1],
    input logic signed [13:0] b_out [0:S-1],
    output logic [31:0] cl [0:N-1]       
    ); 
    
    logic signed [13:0] X_next [0:N-1][0:M-1];
    logic signed [13:0] X_prev [0:N-1][0:M-1];
    logic signed [13:0] X_final [0:N-1][0:M-1];
    
    logic signed [13:0] h[0:M-1]; 
    logic signed [13:0] c[0:M-1]; 
    logic signed [13:0] inputs[0:M2-1]; 
    logic signed [13:0] h2[0:M-1]; 
    logic signed [13:0] c2[0:M-1]; 
    logic signed [13:0] inputs2[0:M2-1]; 
    
    logic signed [13:0] temp1;
    logic signed [13:0] temp2;
    logic signed [13:0] temp3;
    logic signed [13:0] temp4;    
    
    
    typedef enum logic [3:0] {
        IDLE,
        DENSE_1, 
        DENSE_2, 
        DENSE_3, 
        LSTM_INIT, 
        LSTM_LOOP, 
        LSTM_CALL_1, 
        LSTM_CALL_2, 
        LSTM_sTORE, 
        DENSE_OUT, 
        ARGMAX, 
        FINISH
    } fsm_state_t; 
    
    fsm_state_t state, next_state; 
    
    // oop counters 
    
    logic [$clog2(N) - 1:0] i_cnt; 
    logic [$clog2(M) - 1:0] j_cnt;
    logic [$clog2(S) - 1:0] j_s_cnt;
    
    logic start_sig, done_sig;
    
    logic signed [13:0] sign_in, sig_out; 
    
    logic start_lstm1, done_lstm1;
    logic start_lstm2, done_lstm2;
    
    logic signed [13:0] h1_next [0:M-1]; 
    logic signed [13:0] c1_next [0:M-1];
    logic signed [13:0] h2_next [0:M-1];
    logic signed [13:0] c2_next [0:M-1];
    
    //-- Finite State Machine ---
    
    
    
    //===========================================================
    // Module Instantiations 
    //===========================================================
    sigmoid #(.WIDTH(14)) sigmoid_inst (
        .clk(clk), 
        .start(start_sig), 
        .din(sig_in),
        .done(done_sig), 
        .dout(sig_out) 
    ); 
    
    rnn_lstm_cell #(
        .M(M),
        .M2(M2),
        .M4(M4)    
     ) lstm_i_1 (
        .clk(clk), 
        .rst_n(rst_n), 
        .start(start_lstm1), 
        .done(done_lstm1),
        .w_all(w_all), 
        .b_all(b_all), 
        .h_in(h), 
        .c_in(c), 
        .inputs(inputs), 
        .h_out(h1_next), 
        .c_out(c1_next) 
     );
     
    rnn_lstm_cell #(
        .M(M),
        .M2(M2),
        .M4(M4)    
     ) lstm_i_2 (
        .clk(clk), 
        .rst_n(rst_n), 
        .start(start_lstm2), 
        .done(done_lstm2),
        .w_all(w_all2), 
        .b_all(b_all2), 
        .h_in(h2), 
        .c_in(c2), 
        .inputs(inputs2), 
        .h_out(h2_next), 
        .c_out(c2_next) 
     );
endmodule
