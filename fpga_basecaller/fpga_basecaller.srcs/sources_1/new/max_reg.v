`timescale 1ns / 1ps
module max_reg #(parameter N = 16) (
    input clk,
    input ce,
    input [N-1:0] din,
    input rst_m,
    input master_rst,
    output reg [N-1:0] reg_op
    );

    always@(posedge clk) 
    begin
      if(ce) 
        begin
          if(rst_m) begin
             reg_op <=0;
          end
          else begin
             reg_op <= din;
         end
        end
    end
endmodule