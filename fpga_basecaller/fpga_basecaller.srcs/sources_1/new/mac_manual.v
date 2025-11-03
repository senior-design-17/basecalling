//`define FIXED_POINT 1
module mac_manual #(parameter N = 16,parameter Q = 12)(
    input clk,ce,
    input [N-1:0] a,
    input [N-1:0] b,
    input [N-1:0] c,
    input wire sclr,
    output [N-1:0] p,
    input fast_clk, 
	output wire diff
    );
 
`ifdef FIXED_POINT
    wire [N-1:0] mult,add;
    reg [N-1:0] tmp;
    wire ovr;
     (* use_dsp = "no" *)
    qmult #(N,Q) mul (
                .clk(clk),
                .fast_clk (fast_clk),
                .global_rst(sclr),
                .a(a),
                .b(b),
                .q_result(mult),
                .diff(diff),
                .overflow(ovr)
                );
    qadd #(N,Q) add1 (
                .a(mult),
                .b(c),
                .c(add)
                );
     
    always@(posedge clk) begin
               if(ce)
               begin
                   tmp <= add;
               end
           end
           assign p = tmp;
`else
    reg [N-1:0] temp;
    always@(posedge clk)
    begin
        if(ce)
            begin
                temp <= (a*b+c);
            end
    end
    assign p = temp;
 `endif 
endmodule