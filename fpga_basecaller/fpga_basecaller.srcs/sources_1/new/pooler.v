`timescale 1ns / 1ps

module pooler #(
    parameter m = 9'h00c,
    parameter p = 9'h003,
    parameter N = 16,
    parameter Q = 12,
    parameter ptype = 1, //0-> average pooling, 1 -> max pooling
    parameter p_sqr_inv = 16'b0000010000000000 //this parameter is needed in average pooling case where the sum is divided by p**2.
                                               //It needs to be supplied manually and should be equal to (1/p)^2 in whatever the
                                               //(Q,N) format is being used.
    )(
    input clk,
    input ce,
    input master_rst,
    input fast_clk, 
    input [N-1:0] data_in,
    output [N-1:0] data_out,
    output valid_op,               //output signal to indicate the valid output
    output end_op                  //output signal to indicate when all the valid outputs have been  
                                   //produced for that particular input matrix
    );
   
    wire rst_m,load_sr,global_rst;//op_en,pause_ip,
    wire [1:0] sel;
    wire [N-1:0] comp_op;
    wire [N-1:0] sr_op;
    wire [N-1:0] max_reg_op;
    wire [N-1:0] div_op;
    wire ovr;
    wire [N-1:0] mux_out;
    //reg [N-1:0] temp;
   
    control_logic2 #(
        .m(m),
        .p(p)
        ) log(     
            .clk (clk),
            .master_rst(master_rst),
            .ce (ce),
            .sel (sel),
            .rst_m (rst_m),
            .op_en(valid_op),
            .load_sr(load_sr),
            .global_rst (global_rst),
            .end_op (end_op)
      );
    
    comparator2 #(.N(N),.ptype(ptype)) cmp(
        .ce (ce),         
	    .ip1(data_in),
	    .ip2 (mux_out),
	    .comp_op (comp_op)
      );
  
    max_reg #(.N(N)) m1(               
    	.clk (clk),
    	.ce (ce),
	    .din(comp_op),
	    .rst_m (rst_m),
	    .reg_op(max_reg_op)
      );
 
    variable_shift_reg #(.WIDTH(N),.SIZE((m/p))) SR (
         .d(comp_op),                 
         .clk(clk),                 
         .ce(load_sr),                 
         .rst(global_rst),         
         .out(sr_op)             
         );

   input_mux #(
        .N(N)
            ) mux (
            sr_op,  
            max_reg_op,
            sel,
            mux_out
        );
   
   qmult #(N,Q) mul (
        .clk(clk),
        .fast_clk (fast_clk), 
        .global_rst (rst_m), 
        .a (max_reg_op),
        .b (p_sqr_inv), 
        .q_result (div_op),
        .overflow (ovr)
    );
  
   
   assign data_out = ptype ? max_reg_op : div_op; //for average pooling, we output the sum divided by p**2 
endmodule