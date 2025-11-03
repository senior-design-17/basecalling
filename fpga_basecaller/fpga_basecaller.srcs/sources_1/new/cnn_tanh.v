`timescale 1ns / 1ps
`define FIXED_POINT 1
module cnn_tanh #(
    parameter n = 9'd224,  //size of the input image/activation map
    parameter k = 9'h003,  //size of the convolution window
    parameter p = 9'h002,  //size of the pooling window
    parameter s = 1,       //stride value during convolution
    parameter ptype = 1,  //0 => average pooling , 1 => max_pooling
    parameter act_type = 0,//0 => ReLu activation function, 1=> Hyperbolic tangent activation function
    parameter N = 16,     //Bit width of activations and weights (total datapath width)
    parameter Q = 12,     //Number of fractional bits in case of fixed point representation
    parameter AW = 10,    //Needed in case of tanh activation function to set the size or ROM
    parameter DW = 16,    //Datapath width = N 
    parameter p_sqr_inv = 16'b0000010000000000 // = 1/p**2 in the (N,Q) format being used currently
    )(
    input clk,
    input ce,
    input global_rst, 
    input [15:0] activation,
    input [(k*k)*16-1:0] weight1,
    input fast_clk, 
    output [15:0] data_out,
    output valid_op,
    output end_op,
    output [15:0] conv_out,
    output conv_valid, 
    output conv_end
    );
    
    wire [N-1:0] conv_op;
    wire valid_conv,end_conv;
    wire valid_ip;
    wire [N-1:0] relu_op;
    wire [N-1:0] tanh_op;
    wire [N-1:0] pooler_ip;
    wire [N-1:0] pooler_op;
    reg [N-1:0] pooler_op_reg;
    reg [N-1:0] conv_op_pipeline_reg; 
    reg [N-1:0] relu_op_pipeline_reg; 
    reg rst; 
    
    always @(posedge clk) begin 
        rst <= global_rst;
    end
    always @(posedge clk) begin 
        conv_op_pipeline_reg <= conv_op;
        relu_op_pipeline_reg <= relu_op; 
    end
    convolver #(.n(n),.k(k),.s(s),.N(N),.Q(Q)) conv(//Convolution engine
            .clk(clk), 
            .ce(ce), 
            .fast_clk (fast_clk), 
            .global_rst (rst), 
            .weight1(weight1),  
            .activation(activation), 
            .conv_op(conv_op), 
            .end_conv(end_conv), 
            .diff (diff), 
            .valid_conv(valid_conv)
        );
    assign conv_valid = valid_conv;
    assign conv_end = end_conv;
    assign conv_out = conv_op;
    
    assign valid_ip = valid_conv && (!end_conv);
    
tanh_lut #()(
    .clk(clk),
    .rst(rst),
    .phase (phase),
    .tanh(tanh)
    );
    
    
    assign pooler_ip = act_type ? tanh_op : relu_op_pipeline_reg; //alternatively you could use macros to save resources when using ReLu
    
    pooler #(.N(N),.Q(Q),.m(n-k+1),.p(p),.ptype(ptype),.p_sqr_inv(p_sqr_inv)) pool( //Pooling Unit
            .clk(clk),
            .ce(valid_ip),
            .master_rst(rst),
            .fast_clk (fast_clk),
            .data_in(pooler_ip),
            .data_out(pooler_op),
            .valid_op(valid_op),
            .end_op(end_op)
        );

    assign data_out = pooler_op;
    
endmodule