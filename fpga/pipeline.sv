//======================================================================
// MASTER Top-Level Module: End-to-End Sepsis Diagnostic Pipeline
module pod5_to_sepsis_report (
    input wire clk,
    input wire reset_n,
    input wire start_pipeline,
    
    output wire [7:0]  report_char_out,
    output wire        report_valid_out,
    input  wire        report_ready_in,
    
    output wire        sepsis_detected,
    output wire [7:0]  sepsis_confidence,
    output wire [15:0] pathogen_id,
    output wire [7:0]  pathogen_load,
    output wire        results_valid,
    
    output wire uart_tx_pin
);

    localparam CHUNK_SIGNAL_SAMPLES = 10000;
    localparam NN_TIME_STEPS        = 1; //make time stamp 1
    localparam NN_FEATURE_DIM       = 384;
    localparam MAX_READ_LENGTH      = 150;
    localparam NUM_HOST_GENES       = 20000;
    localparam NUM_BIOMARKERS       = 50;
    localparam NUM_PATHOGENS        = 1000;

    // Stage wires
    wire [127:0] s1_read_id;
    wire [15:0]  s1_sample_data;
    wire         s1_is_last_chunk;
    wire         s1_end_of_chunk;
    wire         s1_sample_valid;
    wire         s1_sample_ready;

    wire [127:0] s2_read_id;
    wire [31:0]  s2_feature_data;
    wire         s2_is_last_chunk;
    wire         s2_end_of_chunk;
    wire         s2_feature_valid;
    wire         s2_feature_ready;

    wire [127:0] s3_read_id;
    wire [7:0]   s3_base_char;
    wire [7:0]   s3_qual_char;
    wire         s3_end_of_read;
    wire         s3_base_valid;
    wire         s3_base_ready;

    wire [7:0]   s4_fastq_char;
    wire         s4_end_of_read;
    wire         s4_fastq_valid;
    wire         s4_fastq_ready;

    wire [7:0]   s5_fastq_char;
    wire [127:0] s5_read_id;
    wire         s5_end_of_read;
    wire         s5_fastq_valid;
    wire         s5_fastq_ready_host;
    wire         s5_fastq_ready_path;
    wire         s5_fastq_ready;

    wire [127:0] s6_read_id;
    wire [31:0]  s6_gene_id;
    wire [7:0]   s6_mapping_quality;
    wire         s6_is_host_aligned;
    wire         s6_align_valid;
    wire         s6_align_ready;

    wire [127:0] s6p_read_id;
    wire [15:0]  s6p_pathogen_id;
    wire [7:0]   s6p_mapping_quality;
    wire         s6p_is_pathogen;
    wire         s6p_align_valid;
    wire         s6p_align_ready;

    wire [31:0]  s7_gene_id;
    wire [31:0]  s7_raw_count;
    wire [31:0]  s7_tpm;
    wire         s7_count_valid;
    wire         s7_count_ready;

    wire [15:0]  s8_pathogen_id;
    wire [31:0]  s8_pathogen_count;
    wire [7:0]   s8_pathogen_abundance;
    wire         s8_pathogen_valid;
    wire         s8_pathogen_ready;

    wire [7:0]   s9_biomarker_id;
    wire [31:0]  s9_expression_level;
    wire [7:0]   s9_fold_change;
    wire         s9_biomarker_valid;
    wire         s9_biomarker_ready;

    wire         s10_sepsis_detected;
    wire [7:0]   s10_confidence;
    wire [7:0]   s10_severity_score;
    wire [15:0]  s10_top_pathogen_id;
    wire [7:0]   s10_pathogen_load;
    wire         s10_classification_valid;
    wire         s10_classification_ready;

    //==================================================================
    // STAGE 1: POD5 BRAM Reader - WITH CORRECT CONNECTIONS
    //==================================================================
    pod5_bram_reader_norm #(
        .CHUNK_SIGNAL_SAMPLES(CHUNK_SIGNAL_SAMPLES)
    ) i_stage1_pod5_reader (
        .clk(clk),
        .reset_n(reset_n),
        .start(start_pipeline),
        .start_read_idx(10'd0),      // FIXED: Start from read 0
        .num_reads(10'd1),           // FIXED: Process 1 read
        .o_read_id(s1_read_id),
        .o_sample_data(s1_sample_data),
        .o_is_last_chunk(s1_is_last_chunk),
        .o_end_of_chunk(s1_end_of_chunk),
        .o_sample_valid(s1_sample_valid),
        .i_sample_ready(s1_sample_ready),
        .o_processing_done(),        // Not used
        .o_reads_processed(),        // Not used
        .o_samples_processed()       // Not used
    );

    //==================================================================
    // STAGE 2: Neural Network
    //==================================================================
    fpga_basecaller_nn #(
        .CHUNK_SIGNAL_SAMPLES(CHUNK_SIGNAL_SAMPLES),
        .TIME_STEPS(NN_TIME_STEPS),
        .FEATURE_DIM(NN_FEATURE_DIM)
    ) i_stage2_basecaller_nn (
        .clk(clk),
        .reset_n(reset_n),
        .i_read_id(s1_read_id),
        .i_sample_data(s1_sample_data),
        .i_is_last_chunk(s1_is_last_chunk),
        .i_end_of_chunk(s1_end_of_chunk),
        .i_sample_valid(s1_sample_valid),
        .o_sample_ready(s1_sample_ready),
        .o_read_id(s2_read_id),
        .o_feature_data(s2_feature_data),
        .o_is_last_chunk(s2_is_last_chunk),
        .o_end_of_chunk(s2_end_of_chunk),
        .o_feature_valid(s2_feature_valid),
        .i_feature_ready(s2_feature_ready)
    );

    //==================================================================
    // STAGE 3: Decoder
    //==================================================================
    hw_decoder_crf_beamsearch #(
        .TIME_STEPS(NN_TIME_STEPS),
        .FEATURE_DIM(NN_FEATURE_DIM)
    ) i_stage3_decoder (
        .clk(clk),
        .reset_n(reset_n),
        .i_read_id(s2_read_id),
        .i_feature_data(s2_feature_data),
        .i_is_last_chunk(s2_is_last_chunk),
        .i_end_of_chunk(s2_end_of_chunk),
        .i_feature_valid(s2_feature_valid),
        .o_feature_ready(s2_feature_ready),
        .o_read_id(s3_read_id),
        .o_base_char(s3_base_char),
        .o_qual_char(s3_qual_char),
        .o_end_of_read(s3_end_of_read),
        .o_base_valid(s3_base_valid),
        .i_base_ready(s3_base_ready)
    );

    //==================================================================
    // STAGE 4: FASTQ Formatter
    //==================================================================
    fastq_stitcher_formatter i_stage4_fastq (
        .clk(clk),
        .reset_n(reset_n),
        .i_read_id(s3_read_id),
        .i_base_char(s3_base_char),
        .i_qual_char(s3_qual_char),
        .i_end_of_read(s3_end_of_read),
        .i_base_valid(s3_base_valid),
        .o_base_ready(s3_base_ready),
        .o_fastq_char(s4_fastq_char),
        .o_end_of_read(s4_end_of_read),
        .o_fastq_valid(s4_fastq_valid),
        .i_fastq_ready(s4_fastq_ready)
    );

    //==================================================================
    // STAGE 5: QC Trimmer
    //==================================================================
    fastq_qc_trimmer #(
        .MAX_READ_LENGTH(MAX_READ_LENGTH)
    ) i_stage5_qc (
        .clk(clk),
        .reset_n(reset_n),
        .start(1'b0),
        .i_fastq_char_in(s4_fastq_char),
        .i_end_of_read_in(s4_end_of_read),
        .i_fastq_valid_in(s4_fastq_valid),
        .o_fastq_ready_out(s4_fastq_ready),
        .o_fastq_char(s5_fastq_char),
        .o_read_id(s5_read_id),
        .o_end_of_read(s5_end_of_read),
        .o_fastq_valid(s5_fastq_valid),
        .i_fastq_ready(s5_fastq_ready)
    );

    assign s5_fastq_ready = s5_fastq_ready_host & s5_fastq_ready_path;

    //==================================================================
    // STAGE 6A: Host Aligner
    //==================================================================
    transcriptome_aligner_host #(
        .MAX_READ_LENGTH(MAX_READ_LENGTH),
        .NUM_HOST_GENES(NUM_HOST_GENES)
    ) i_stage6a_host_aligner (
        .clk(clk),
        .reset_n(reset_n),
        .i_fastq_char(s5_fastq_char),
        .i_read_id(s5_read_id),
        .i_end_of_read(s5_end_of_read),
        .i_fastq_valid(s5_fastq_valid),
        .o_fastq_ready(s5_fastq_ready_host),
        .o_read_id(s6_read_id),
        .o_gene_id(s6_gene_id),
        .o_mapping_quality(s6_mapping_quality),
        .o_is_host_aligned(s6_is_host_aligned),
        .o_align_valid(s6_align_valid),
        .i_align_ready(s6_align_ready)
    );

    //==================================================================
    // STAGE 6B: Pathogen Aligner
    //==================================================================
    pathogen_aligner #(
        .MAX_READ_LENGTH(MAX_READ_LENGTH),
        .NUM_PATHOGENS(NUM_PATHOGENS)
    ) i_stage6b_pathogen_aligner (
        .clk(clk),
        .reset_n(reset_n),
        .i_fastq_char(s5_fastq_char),
        .i_read_id(s5_read_id),
        .i_end_of_read(s5_end_of_read),
        .i_fastq_valid(s5_fastq_valid),
        .o_fastq_ready(s5_fastq_ready_path),
        .o_read_id(s6p_read_id),
        .o_pathogen_id(s6p_pathogen_id),
        .o_mapping_quality(s6p_mapping_quality),
        .o_is_pathogen(s6p_is_pathogen),
        .o_align_valid(s6p_align_valid),
        .i_align_ready(s6p_align_ready)
    );

    //==================================================================
    // STAGE 7: Gene Expression Quantifier
    //==================================================================
    gene_expression_quantifier #(
        .NUM_HOST_GENES(NUM_HOST_GENES)
    ) i_stage7_expression (
        .clk(clk),
        .reset_n(reset_n),
        .i_read_id(s6_read_id),
        .i_gene_id(s6_gene_id),
        .i_mapping_quality(s6_mapping_quality),
        .i_is_host_aligned(s6_is_host_aligned),
        .i_align_valid(s6_align_valid),
        .o_align_ready(s6_align_ready),
        .o_gene_id(s7_gene_id),
        .o_raw_count(s7_raw_count),
        .o_tpm(s7_tpm),
        .o_count_valid(s7_count_valid),
        .i_count_ready(s7_count_ready)
    );

    //==================================================================
    // STAGE 8: Pathogen Quantifier
    //==================================================================
    pathogen_quantifier #(
        .NUM_PATHOGENS(NUM_PATHOGENS)
    ) i_stage8_pathogen_quant (
        .clk(clk),
        .reset_n(reset_n),
        .i_read_id(s6p_read_id),
        .i_pathogen_id(s6p_pathogen_id),
        .i_mapping_quality(s6p_mapping_quality),
        .i_is_pathogen(s6p_is_pathogen),
        .i_align_valid(s6p_align_valid),
        .o_align_ready(s6p_align_ready),
        .o_pathogen_id(s8_pathogen_id),
        .o_pathogen_count(s8_pathogen_count),
        .o_pathogen_abundance(s8_pathogen_abundance),
        .o_pathogen_valid(s8_pathogen_valid),
        .i_pathogen_ready(s8_pathogen_ready)
    );

    //==================================================================
    // STAGE 9: Biomarker Analyzer
    //==================================================================
    biomarker_analyzer #(
        .NUM_BIOMARKERS(NUM_BIOMARKERS)
    ) i_stage9_biomarkers (
        .clk(clk),
        .reset_n(reset_n),
        .i_gene_id(s7_gene_id),
        .i_raw_count(s7_raw_count),
        .i_tpm(s7_tpm),
        .i_count_valid(s7_count_valid),
        .o_count_ready(s7_count_ready),
        .o_biomarker_id(s9_biomarker_id),
        .o_expression_level(s9_expression_level),
        .o_fold_change(s9_fold_change),
        .o_biomarker_valid(s9_biomarker_valid),
        .i_biomarker_ready(s9_biomarker_ready)
    );

    //==================================================================
    // STAGE 10: Sepsis Classifier
    //==================================================================
    sepsis_ml_classifier #(
        .NUM_BIOMARKERS(NUM_BIOMARKERS)
    ) i_stage10_classifier (
        .clk(clk),
        .reset_n(reset_n),
        .i_biomarker_id(s9_biomarker_id),
        .i_expression_level(s9_expression_level),
        .i_fold_change(s9_fold_change),
        .i_biomarker_valid(s9_biomarker_valid),
        .o_biomarker_ready(s9_biomarker_ready),
        .i_pathogen_id(s8_pathogen_id),
        .i_pathogen_count(s8_pathogen_count),
        .i_pathogen_abundance(s8_pathogen_abundance),
        .i_pathogen_valid(s8_pathogen_valid),
        .o_pathogen_ready(s8_pathogen_ready),
        .o_sepsis_detected(s10_sepsis_detected),
        .o_confidence(s10_confidence),
        .o_severity_score(s10_severity_score),
        .o_top_pathogen_id(s10_top_pathogen_id),
        .o_pathogen_load(s10_pathogen_load),
        .o_classification_valid(s10_classification_valid),
        .i_classification_ready(s10_classification_ready)
    );

    //==================================================================
    // STAGE 11: Report Generator
    //==================================================================
    clinical_report_formatter i_stage11_report (
        .clk(clk),
        .reset_n(reset_n),
        .i_sepsis_detected(s10_sepsis_detected),
        .i_confidence(s10_confidence),
        .i_severity_score(s10_severity_score),
        .i_top_pathogen_id(s10_top_pathogen_id),
        .i_pathogen_load(s10_pathogen_load),
        .i_classification_valid(s10_classification_valid),
        .o_classification_ready(s10_classification_ready),
        .o_report_char(report_char_out),
        .o_report_valid(report_valid_out),
        .i_report_ready(report_ready_in)
    );

    //==================================================================
    // Parallel Output Registers
    //==================================================================
    reg        sepsis_detected_reg;
    reg [7:0]  sepsis_confidence_reg;
    reg [15:0] pathogen_id_reg;
    reg [7:0]  pathogen_load_reg;
    reg        results_valid_reg;

    assign sepsis_detected   = sepsis_detected_reg;
    assign sepsis_confidence = sepsis_confidence_reg;
    assign pathogen_id       = pathogen_id_reg;
    assign pathogen_load     = pathogen_load_reg;
    assign results_valid     = results_valid_reg;

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            sepsis_detected_reg   <= 1'b0;
            sepsis_confidence_reg <= 8'd0;
            pathogen_id_reg       <= 16'd0;
            pathogen_load_reg     <= 8'd0;
            results_valid_reg     <= 1'b0;
        end else if (s10_classification_valid && s10_classification_ready) begin
            sepsis_detected_reg   <= s10_sepsis_detected;
            sepsis_confidence_reg <= s10_confidence;
            pathogen_id_reg       <= s10_top_pathogen_id;
            pathogen_load_reg     <= s10_pathogen_load;
            results_valid_reg     <= 1'b1;
        end
    end
    
    assign uart_tx_pin = 1'b1; // Placeholder

endmodule

//======================================================================
// STAGE 1: POD5 BRAM Reader & Normalizer
// Production-Ready Implementation
//======================================================================
// This module reads raw signal data from Oxford Nanopore POD5 format
// stored in BRAM and streams out normalized samples.
//
// POD5 Format (Simplified for FPGA):
// - Each read has metadata: read_id, num_samples, num_chunks
// - Signal data: 16-bit ADC values (raw electrical current)
// - Normalization: (raw - median) / MAD (Median Absolute Deviation)
//
// Features:
// - Dual-port BRAM for metadata and signal storage
// - Streaming normalization (median/MAD pre-computed, stored in metadata)
// - AXI-Stream compliant valid/ready handshaking
// - Handles multiple reads sequentially
// - Configurable chunk size for downstream processing
//======================================================================

module pod5_bram_reader_norm #(
    parameter CHUNK_SIGNAL_SAMPLES = 10000,  // Samples per chunk
    parameter MAX_READS            = 1024,   // Max reads in BRAM
    parameter MAX_SAMPLES_PER_READ = 100000, // Max samples per read
    
    // BRAM addressing
    parameter METADATA_ADDR_WIDTH  = 10,     // 2^10 = 1024 reads
    parameter SIGNAL_ADDR_WIDTH    = 17      // 2^17 = 131k samples
) (
    input wire clk,
    input wire reset_n,
    
    // Control
    input wire start,                        // Begin processing reads from BRAM
    
    // Configuration (optional - for advanced use)
    input wire [METADATA_ADDR_WIDTH-1:0] start_read_idx,  // Which read to start from
    input wire [METADATA_ADDR_WIDTH-1:0] num_reads,       // How many reads to process
    
    // Streaming Output Interface (AXI-Stream-like)
    output logic [127:0] o_read_id,          // UUID for this read
    output logic [15:0]  o_sample_data,      // Normalized 16-bit sample
    output logic         o_is_last_chunk,    // 1 = last chunk of this read
    output logic         o_end_of_chunk,     // 1 = last sample of this chunk (TLAST)
    output logic         o_sample_valid,     // 1 = output data is valid
    input  wire          i_sample_ready,     // 1 = downstream ready to receive
    
    // Status outputs
    output logic         o_processing_done,  // All reads processed
    output logic [31:0]  o_reads_processed,  // Counter
    output logic [31:0]  o_samples_processed // Counter
);

    //==================================================================
    // BRAM Structures
    //==================================================================
    
    // --- Metadata BRAM ---
    // Each entry stores per-read metadata
    typedef struct packed {
        logic [127:0] read_id;              // UUID
        logic [31:0]  signal_start_addr;    // Starting address in signal BRAM
        logic [31:0]  num_samples;          // Total samples for this read
        logic [15:0]  median;               // Pre-computed median (for normalization)
        logic [15:0]  mad;                  // Pre-computed MAD (median abs deviation)
        logic [15:0]  offset;               // Calibration offset
        logic [15:0]  scale;                // Calibration scale (fixed-point)
    } metadata_t;
    
    metadata_t metadata_bram [0:(1<<METADATA_ADDR_WIDTH)-1];
    
    // --- Signal BRAM ---
    // Stores raw 16-bit ADC samples
    logic [15:0] signal_bram [0:(1<<SIGNAL_ADDR_WIDTH)-1];
    
    // BRAM read ports
    metadata_t current_metadata;
    logic [15:0] raw_signal;
    
    //==================================================================
    // FSM States
    //==================================================================
    typedef enum logic [2:0] {
        IDLE            = 3'd0,  // Waiting for start
        LOAD_METADATA   = 3'd1,  // Reading metadata for current read
        STREAM_CHUNK    = 3'd2,  // Streaming samples for one chunk
        ADVANCE_READ    = 3'd3,  // Move to next read
        DONE            = 3'd4   // All reads processed
    } state_t;
    
    state_t state, next_state;
    
    //==================================================================
    // Registers
    //==================================================================
    
    // Read management
    logic [METADATA_ADDR_WIDTH-1:0] current_read_idx;
    logic [METADATA_ADDR_WIDTH-1:0] total_reads_to_process;
    logic [31:0] reads_processed_count;
    
    // Sample management within current read
    logic [31:0] current_sample_idx;      // Index within current read (0 to num_samples-1)
    logic [31:0] samples_in_current_chunk; // How many samples in this chunk (0 to CHUNK_SIGNAL_SAMPLES-1)
    logic [31:0] total_samples_processed;
    
    // Signal BRAM addressing
    logic [SIGNAL_ADDR_WIDTH-1:0] signal_read_addr;
    
    // Normalization pipeline
    logic signed [31:0] norm_intermediate;   // (raw - median)
    logic signed [15:0] normalized_sample;   // Final normalized output
    
    // Pipeline valid flags
    logic bram_read_valid;
    logic norm_pipe_valid;
    
    //==================================================================
    // BRAM Read Logic (Synchronous)
    //==================================================================
    
    always_ff @(posedge clk) begin
        if (state == LOAD_METADATA) begin
            current_metadata <= metadata_bram[current_read_idx];
        end
        
        if (state == STREAM_CHUNK && (!o_sample_valid || i_sample_ready)) begin
            raw_signal <= signal_bram[signal_read_addr];
            bram_read_valid <= 1'b1;
        end else begin
            bram_read_valid <= 1'b0;
        end
    end
    
    //==================================================================
    // Normalization Pipeline (2-stage)
    //==================================================================
    // Stage 1: Subtract median
    // Stage 2: Divide by MAD (using pre-computed reciprocal)
    
    always_ff @(posedge clk) begin
        if (!reset_n) begin
            norm_intermediate <= '0;
            normalized_sample <= '0;
            norm_pipe_valid   <= 1'b0;
        end else begin
            // Stage 1: (raw - median)
            if (bram_read_valid) begin
                norm_intermediate <= $signed(raw_signal) - $signed(current_metadata.median);
                norm_pipe_valid   <= 1'b1;
            end else begin
                norm_pipe_valid <= 1'b0;
            end
            
            // Stage 2: Divide by MAD
            // Simplified: multiply by reciprocal (MAD stored as fixed-point reciprocal)
            if (norm_pipe_valid) begin
                // Fixed-point multiply: (value * scale) >> 8
                // Scale is stored as (1.0 / MAD) * 256 for fixed-point math
                normalized_sample <= (norm_intermediate * $signed(current_metadata.mad)) >>> 8;
            end
        end
    end
    
    //==================================================================
    // Output Assignment
    //==================================================================
    
    assign o_read_id       = current_metadata.read_id;
    assign o_sample_data   = normalized_sample;
    assign o_processing_done = (state == DONE);
    assign o_reads_processed = reads_processed_count;
    assign o_samples_processed = total_samples_processed;
    
    //==================================================================
    // FSM - Combinational Logic
    //==================================================================
    
    always_comb begin
        // Defaults
        next_state        = state;
        o_sample_valid    = 1'b0;
        o_end_of_chunk    = 1'b0;
        o_is_last_chunk   = 1'b0;
        
        case (state)
            IDLE: begin
                if (start) begin
                    next_state = LOAD_METADATA;
                end
            end
            
            LOAD_METADATA: begin
                // Wait 1 cycle for BRAM read
                next_state = STREAM_CHUNK;
            end
            
            STREAM_CHUNK: begin
                // Output is valid when normalization pipeline has data
                o_sample_valid = norm_pipe_valid;
                
                // Determine if this is the last sample in the chunk
                if (samples_in_current_chunk == CHUNK_SIGNAL_SAMPLES - 1) begin
                    o_end_of_chunk = 1'b1;
                end
                
                // Determine if this is the last chunk of the read
                if (current_sample_idx >= current_metadata.num_samples - 1) begin
                    o_is_last_chunk = 1'b1;
                end
                
                // Advance when downstream accepts data
                if (norm_pipe_valid && i_sample_ready) begin
                    if (current_sample_idx >= current_metadata.num_samples - 1) begin
                        // Read complete
                        next_state = ADVANCE_READ;
                    end else if (samples_in_current_chunk == CHUNK_SIGNAL_SAMPLES - 1) begin
                        // Chunk complete, but more samples in read
                        // Stay in STREAM_CHUNK but will reset chunk counter
                        next_state = STREAM_CHUNK;
                    end
                    // else: stay in STREAM_CHUNK
                end
            end
            
            ADVANCE_READ: begin
                if (current_read_idx >= total_reads_to_process - 1) begin
                    next_state = DONE;
                end else begin
                    next_state = LOAD_METADATA;
                end
            end
            
            DONE: begin
                next_state = DONE; // Stay here until reset
            end
            
            default: begin
                next_state = IDLE;
            end
        endcase
    end
    
    //==================================================================
    // FSM - Sequential Logic
    //==================================================================
    
    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            state                     <= IDLE;
            current_read_idx          <= '0;
            current_sample_idx        <= '0;
            samples_in_current_chunk  <= '0;
            reads_processed_count     <= '0;
            total_samples_processed   <= '0;
            signal_read_addr          <= '0;
            total_reads_to_process    <= '0;
        end else begin
            state <= next_state;
            
            case (state)
                IDLE: begin
                    if (start) begin
                        current_read_idx         <= start_read_idx;
                        total_reads_to_process   <= num_reads;
                        current_sample_idx       <= '0;
                        samples_in_current_chunk <= '0;
                        reads_processed_count    <= '0;
                        total_samples_processed  <= '0;
                    end
                end
                
                LOAD_METADATA: begin
                    // Prepare to read first sample
                    signal_read_addr         <= current_metadata.signal_start_addr[SIGNAL_ADDR_WIDTH-1:0];
                    current_sample_idx       <= '0;
                    samples_in_current_chunk <= '0;
                end
                
                STREAM_CHUNK: begin
                    if (norm_pipe_valid && i_sample_ready) begin
                        // Advance sample pointer
                        current_sample_idx       <= current_sample_idx + 1;
                        signal_read_addr         <= signal_read_addr + 1;
                        total_samples_processed  <= total_samples_processed + 1;
                        
                        // Manage chunk counter
                        if (samples_in_current_chunk == CHUNK_SIGNAL_SAMPLES - 1) begin
                            samples_in_current_chunk <= '0;
                        end else begin
                            samples_in_current_chunk <= samples_in_current_chunk + 1;
                        end
                    end
                end
                
                ADVANCE_READ: begin
                    current_read_idx      <= current_read_idx + 1;
                    reads_processed_count <= reads_processed_count + 1;
                end
                
                DONE: begin
                    // Stay idle
                end
            endcase
        end
    end
   
endmodule

//======================================================================
// STAGE 2: FPGA Basecaller Neural Network (CNN + LSTM)

module fpga_basecaller_nn #(
    parameter CHUNK_SIGNAL_SAMPLES = 10000,  // Input samples (changed from 10000
    parameter TIME_STEPS           = 1,   // Output time steps //make 1
    parameter FEATURE_DIM          = 8,    // Features per time step (384/20)
    
    // Network architecture
    parameter CNN_LAYERS           = 7,
    parameter LSTM_LAYERS          = 6,
    parameter LSTM_HIDDEN_SIZE     = 1, //384 TO 1
    
    // Quantization
    parameter WEIGHT_WIDTH         = 8,      // INT8 weights
    parameter ACTIVATION_WIDTH     = 4,     // INT16 activations
    parameter ACCUMULATOR_WIDTH    = 32      // INT32 accumulators
) (
    input wire clk,
    input wire reset_n,
    
    // --- Streaming Input (from Stage 1) ---
    input wire [127:0] i_read_id,
    input wire [15:0]  i_sample_data,
    input wire         i_is_last_chunk,
    input wire         i_end_of_chunk,
    input wire         i_sample_valid,
    output logic       o_sample_ready,
    
    // --- Streaming Output (to Stage 3) ---
    output logic [127:0] o_read_id,
    output logic [31:0]  o_feature_data,      // Float32 probability
    output logic         o_is_last_chunk,
    output logic         o_end_of_chunk,
    output logic         o_feature_valid,
    input  wire          i_feature_ready
);

    //==================================================================
    // FSM States
    //==================================================================
    typedef enum logic [2:0] {
        IDLE            = 3'd0,  // Waiting for input
        RECEIVE_SIGNAL  = 3'd1,  // Buffering input samples
        CNN_PROCESS     = 3'd2,  // Running CNN frontend
        LSTM_PROCESS    = 3'd3,  // Running LSTM backend
        STREAM_OUTPUT   = 3'd4   // Streaming features to Stage 3
    } state_t;
    
    state_t state, next_state;
    
    //==================================================================
    // Internal Buffers
    //==================================================================
    
    // Input buffer (stores 10k samples for batch processing)
    logic [15:0] input_buffer [0:CHUNK_SIGNAL_SAMPLES-1];
    logic [31:0] input_sample_count;
    
    // CNN output buffer (intermediate features)
    // After CNN: 10000 samples → 1667 frames (stride=6)
    logic [ACTIVATION_WIDTH-1:0] cnn_features [0:TIME_STEPS-1][0:127];
    
    // LSTM output buffer (final features)
    logic [31:0] lstm_features [0:TIME_STEPS-1][0:FEATURE_DIM-1];
    
    // Output streaming state
    logic [31:0] output_timestep;
    logic [31:0] output_feature_idx;
    
    // Pipeline metadata
    logic [127:0] read_id_reg;
    logic         is_last_chunk_reg;
    
    //==================================================================
    // Weight BRAMs (Pre-loaded during configuration)
    //==================================================================
    
    // CNN weights (5 layers, varying sizes)
    // Layer 0: 1 input channel, 16 output channels, kernel=5
    // Layer 1-4: Progressive channel expansion
    logic signed [WEIGHT_WIDTH-1:0] cnn_weights_l0 [0:16*1*5-1];
    logic signed [WEIGHT_WIDTH-1:0] cnn_weights_l1 [0:32*16*5-1];
    logic signed [WEIGHT_WIDTH-1:0] cnn_weights_l2 [0:64*32*5-1];
    logic signed [WEIGHT_WIDTH-1:0] cnn_weights_l3 [0:128*64*5-1];
    logic signed [WEIGHT_WIDTH-1:0] cnn_weights_l4 [0:128*128*3-1];
    
    logic signed [WEIGHT_WIDTH-1:0] lstm_weights [0:LSTM_LAYERS-1][0:2-1][0:4*LSTM_HIDDEN_SIZE*LSTM_HIDDEN_SIZE-1];
    
    //==================================================================
    // CNN Processing Pipeline
    //==================================================================
    
    // CNN Submodule (encapsulates all 5 layers)
    logic        cnn_start;
    logic        cnn_done;
    logic [31:0] cnn_progress; // For monitoring
    
    cnn_frontend #(
        .INPUT_LENGTH(CHUNK_SIGNAL_SAMPLES),
        .OUTPUT_LENGTH(TIME_STEPS),
        .NUM_LAYERS(CNN_LAYERS),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACTIVATION_WIDTH(ACTIVATION_WIDTH)
    ) cnn_frontend_inst (
        .clk(clk),
        .reset_n(reset_n),
        .i_start(cnn_start),
        .i_input_buffer(input_buffer),
        .o_features(cnn_features),
        .o_done(cnn_done),
        .o_progress(cnn_progress),
        // Weight interfaces
        .i_weights_l0(cnn_weights_l0),
        .i_weights_l1(cnn_weights_l1),
        .i_weights_l2(cnn_weights_l2),
        .i_weights_l3(cnn_weights_l3),
        .i_weights_l4(cnn_weights_l4)
    );
    
    //==================================================================
    // LSTM Processing Pipeline
    //==================================================================
    
    // LSTM Submodule (3 bidirectional layers)
    logic        lstm_start;
    logic        lstm_done;
    logic [31:0] lstm_progress;
    
    lstm_backend #(
        .TIME_STEPS(TIME_STEPS),
        .INPUT_DIM(128),           // From CNN output
        .HIDDEN_SIZE(LSTM_HIDDEN_SIZE),
        .NUM_LAYERS(LSTM_LAYERS),
        .OUTPUT_DIM(FEATURE_DIM),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACTIVATION_WIDTH(ACTIVATION_WIDTH)
    ) lstm_backend_inst (
        .clk(clk),
        .reset_n(reset_n),
        .i_start(lstm_start),
        .i_features(cnn_features),
        .o_output(lstm_features),
        .o_done(lstm_done),
        .o_progress(lstm_progress),
        // Weight interface
        .i_weights(lstm_weights)
    );
    
    //==================================================================
    // FSM - Combinational Logic
    //==================================================================
    
    always_comb begin
        // Defaults
        next_state     = state;
        o_sample_ready = 1'b0;
        o_feature_valid = 1'b0;
        o_end_of_chunk = 1'b0;
        cnn_start      = 1'b0;
        lstm_start     = 1'b0;
        
        case (state)
            IDLE: begin
                o_sample_ready = 1'b1;
                if (i_sample_valid) begin
                    next_state = RECEIVE_SIGNAL;
                end
            end
            
            RECEIVE_SIGNAL: begin
                o_sample_ready = 1'b1;
                if (i_sample_valid && i_end_of_chunk) begin
                    // All samples received
                    next_state = CNN_PROCESS;
                end
            end
            
            CNN_PROCESS: begin
                if (input_sample_count == 1) begin
                    // Trigger CNN on first cycle
                    cnn_start = 1'b1;
                end
                
                if (cnn_done) begin
                    next_state = LSTM_PROCESS;
                end
            end
            
            LSTM_PROCESS: begin
                if (input_sample_count == 1) begin
                    // Trigger LSTM on first cycle
                    lstm_start = 1'b1;
                end
                
                if (lstm_done) begin
                    next_state = STREAM_OUTPUT;
                end
            end
            
            STREAM_OUTPUT: begin
                o_feature_valid = 1'b1;
                
                // Signal end of chunk when all features streamed
                if (output_timestep == TIME_STEPS - 1 && 
                    output_feature_idx == FEATURE_DIM - 1) begin
                    o_end_of_chunk = 1'b1;
                end
                
                if (i_feature_ready) begin
                    if (output_timestep == TIME_STEPS - 1 && 
                        output_feature_idx == FEATURE_DIM - 1) begin
                        // All features sent
                        next_state = IDLE;
                    end
                end
            end
            
            default: begin
                next_state = IDLE;
            end
        endcase
    end
    
    //==================================================================
    // FSM - Sequential Logic
    //==================================================================
    
    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            state               <= IDLE;
            input_sample_count  <= '0;
            output_timestep     <= '0;
            output_feature_idx  <= '0;
            read_id_reg         <= '0;
            is_last_chunk_reg   <= '0;
        end else begin
            state <= next_state;
            
            case (state)
                IDLE: begin
                    input_sample_count <= '0;
                    output_timestep    <= '0;
                    output_feature_idx <= '0;
                end
                
                RECEIVE_SIGNAL: begin
                    if (i_sample_valid && o_sample_ready) begin
                        // Store sample in buffer
                        input_buffer[input_sample_count] <= i_sample_data;
                        input_sample_count <= input_sample_count + 1;
                        
                        // Latch metadata on first sample
                        if (input_sample_count == 0) begin
                            read_id_reg       <= i_read_id;
                            is_last_chunk_reg <= i_is_last_chunk;
                        end
                    end
                end
                
                CNN_PROCESS: begin
                    // Wait for CNN to complete
                    input_sample_count <= '0; // Reset for next use
                end
                
                LSTM_PROCESS: begin
                    // Wait for LSTM to complete
                end
                
                STREAM_OUTPUT: begin
                    if (o_feature_valid && i_feature_ready) begin
                        // Advance to next feature
                        if (output_feature_idx == FEATURE_DIM - 1) begin
                            output_feature_idx <= '0;
                            output_timestep    <= output_timestep + 1;
                        end else begin
                            output_feature_idx <= output_feature_idx + 1;
                        end
                    end
                end
            endcase
        end
    end
    
    //==================================================================
    // Output Assignment
    //==================================================================
    
    assign o_read_id       = read_id_reg;
    assign o_is_last_chunk = is_last_chunk_reg;
    assign o_feature_data  = lstm_features[output_timestep][output_feature_idx];

endmodule

//======================================================================
// CNN Frontend Submodule
//======================================================================
module cnn_frontend #(
    parameter INPUT_LENGTH     = 10000,
    parameter OUTPUT_LENGTH    = 1667,
    parameter NUM_LAYERS       = 5,
    parameter WEIGHT_WIDTH     = 8,
    parameter ACTIVATION_WIDTH = 16
) (
    input wire clk,
    input wire reset_n,
    input wire i_start,
    
    // Input buffer
    input wire [15:0] i_input_buffer [0:INPUT_LENGTH-1],
    
    // Output features
    output logic [ACTIVATION_WIDTH-1:0] o_features [0:OUTPUT_LENGTH-1][0:127],
    
    // Status
    output logic        o_done,
    output logic [31:0] o_progress,
    
    // Weights (pre-loaded)
    input wire signed [WEIGHT_WIDTH-1:0] i_weights_l0 [0:16*1*5-1],
    input wire signed [WEIGHT_WIDTH-1:0] i_weights_l1 [0:32*16*5-1],
    input wire signed [WEIGHT_WIDTH-1:0] i_weights_l2 [0:64*32*5-1],
    input wire signed [WEIGHT_WIDTH-1:0] i_weights_l3 [0:128*64*5-1],
    input wire signed [WEIGHT_WIDTH-1:0] i_weights_l4 [0:128*128*3-1]
);

    // FSM for CNN processing
    typedef enum logic [2:0] {
        CNN_IDLE   = 3'd0,
        CNN_LAYER0 = 3'd1,
        CNN_LAYER1 = 3'd2,
        CNN_LAYER2 = 3'd3,
        CNN_LAYER3 = 3'd4,
        CNN_LAYER4 = 3'd5,
        CNN_DONE   = 3'd6
    } cnn_state_t;
    
    cnn_state_t cnn_state;
    
    // Intermediate feature maps
    logic [ACTIVATION_WIDTH-1:0] layer0_out [0:9999][0:15];   // Stride=1
    logic [ACTIVATION_WIDTH-1:0] layer1_out [0:4999][0:31];   // Stride=2
    logic [ACTIVATION_WIDTH-1:0] layer2_out [0:2499][0:63];   // Stride=2
    logic [ACTIVATION_WIDTH-1:0] layer3_out [0:1666][0:127];  // Stride=1.5 (approx)
    
    // Processing counters
    logic [31:0] process_idx;
    
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            cnn_state  <= CNN_IDLE;
            o_done     <= 1'b0;
            o_progress <= '0;
            process_idx <= '0;
        end else begin
            case (cnn_state)
                CNN_IDLE: begin
                    o_done <= 1'b0;
                    if (i_start) begin
                        cnn_state   <= CNN_LAYER0;
                        process_idx <= '0;
                    end
                end
                
                CNN_LAYER0: begin
                    // Conv1D: kernel=5, stride=1, channels=1→16
                    if (process_idx < INPUT_LENGTH) begin
                        // Perform convolution at current position
                        // (Simplified: real implementation uses systolic array)
                        for (int c = 0; c < 16; c++) begin
                            logic signed [31:0] acc = 0;
                            for (int k = 0; k < 5; k++) begin
                                if (process_idx + k < INPUT_LENGTH) begin
                                    acc += $signed(i_input_buffer[process_idx + k]) * 
                                           i_weights_l0[c*5 + k];
                                end
                            end
                            // ReLU activation
                            layer0_out[process_idx][c] <= (acc > 0) ? acc[ACTIVATION_WIDTH-1:0] : 0;
                        end
                        process_idx <= process_idx + 1;
                    end else begin
                        cnn_state   <= CNN_LAYER1;
                        process_idx <= '0;
                    end
                    o_progress <= (process_idx * 100) / INPUT_LENGTH;
                end
                
                CNN_LAYER1: begin
                    // Conv1D: kernel=5, stride=2, channels=16→32
                    if (process_idx < 5000) begin
                        // Similar convolution logic (omitted for brevity)
                        process_idx <= process_idx + 1;
                    end else begin
                        cnn_state   <= CNN_LAYER2;
                        process_idx <= '0;
                    end
                end
                
                CNN_LAYER2: begin
                    // Conv1D: kernel=5, stride=2, channels=32→64
                    if (process_idx < 2500) begin
                        process_idx <= process_idx + 1;
                    end else begin
                        cnn_state   <= CNN_LAYER3;
                        process_idx <= '0;
                    end
                end
                
                CNN_LAYER3: begin
                    // Conv1D: kernel=5, stride=1.5, channels=64→128
                    if (process_idx < OUTPUT_LENGTH) begin
                        process_idx <= process_idx + 1;
                    end else begin
                        cnn_state   <= CNN_LAYER4;
                        process_idx <= '0;
                    end
                end
                
                CNN_LAYER4: begin
                    // Conv1D: kernel=3, stride=1, channels=128→128
                    if (process_idx < OUTPUT_LENGTH) begin
                        // Final layer outputs to o_features
                        for (int c = 0; c < 128; c++) begin
                            logic signed [31:0] acc = 0;
                            for (int k = 0; k < 3; k++) begin
                                if (process_idx + k < OUTPUT_LENGTH) begin
                                    acc += $signed(layer3_out[process_idx + k][c]) * 
                                           i_weights_l4[c*3 + k];
                                end
                            end
                            o_features[process_idx][c] <= (acc > 0) ? acc[ACTIVATION_WIDTH-1:0] : 0;
                        end
                        process_idx <= process_idx + 1;
                    end else begin
                        cnn_state <= CNN_DONE;
                    end
                end
                
                CNN_DONE: begin
                    o_done    <= 1'b1;
                    o_progress <= 100;
                    cnn_state <= CNN_IDLE; // Ready for next chunk
                end
            endcase
        end
    end

endmodule

//======================================================================
// LSTM Backend Submodule
//======================================================================
module lstm_backend #(
    parameter TIME_STEPS       = 1667,
    parameter INPUT_DIM        = 128,
    parameter HIDDEN_SIZE      = 384,
    parameter NUM_LAYERS       = 3,
    parameter OUTPUT_DIM       = 384,
    parameter WEIGHT_WIDTH     = 8,
    parameter ACTIVATION_WIDTH = 16
) (
    input wire clk,
    input wire reset_n,
    input wire i_start,
    
    // Input features from CNN
    input wire [ACTIVATION_WIDTH-1:0] i_features [0:TIME_STEPS-1][0:INPUT_DIM-1],
    
    // Output probabilities
    output logic [31:0] o_output [0:TIME_STEPS-1][0:OUTPUT_DIM-1],
    
    // Status
    output logic        o_done,
    output logic [31:0] o_progress,
    
    // Weights
    input wire signed [WEIGHT_WIDTH-1:0] i_weights [0:NUM_LAYERS-1][0:1][0:4*HIDDEN_SIZE*HIDDEN_SIZE-1]
);

    typedef enum logic [2:0] {
        LSTM_IDLE     = 3'd0,
        LSTM_FORWARD  = 3'd1,  // Forward pass (left-to-right)
        LSTM_BACKWARD = 3'd2,  // Backward pass (right-to-left)
        LSTM_COMBINE  = 3'd3,  // Combine bidirectional outputs
        LSTM_SOFTMAX  = 3'd4,  // Apply softmax per timestep
        LSTM_DONE     = 3'd5
    } lstm_state_t;
    
    lstm_state_t lstm_state;
    
    // Hidden states for each layer
    logic [31:0] h_forward  [0:NUM_LAYERS-1][0:TIME_STEPS-1][0:HIDDEN_SIZE-1];
    logic [31:0] h_backward [0:NUM_LAYERS-1][0:TIME_STEPS-1][0:HIDDEN_SIZE-1];
    logic [31:0] c_forward  [0:NUM_LAYERS-1][0:TIME_STEPS-1][0:HIDDEN_SIZE-1]; // Cell state
    logic [31:0] c_backward [0:NUM_LAYERS-1][0:TIME_STEPS-1][0:HIDDEN_SIZE-1];
    
    logic [31:0] timestep;
    logic [31:0] layer_idx;
    
    always_ff @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            lstm_state <= LSTM_IDLE;
            o_done     <= 1'b0;
            timestep   <= '0;
            layer_idx  <= '0;
        end else begin
            case (lstm_state)
                LSTM_IDLE: begin
                    if (i_start) begin
                        lstm_state <= LSTM_FORWARD;
                        timestep   <= '0;
                        layer_idx  <= '0;
                    end
                end
                
                LSTM_FORWARD: begin
                    // Process one timestep for current layer
                    if (timestep < TIME_STEPS) begin
                        // LSTM cell computation (simplified)
                        // Real: i_t = σ(W_i * [h_{t-1}, x_t])
                        //       f_t = σ(W_f * [h_{t-1}, x_t])
                        //       o_t = σ(W_o * [h_{t-1}, x_t])
                        //       c_t = f_t * c_{t-1} + i_t * tanh(W_c * [h_{t-1}, x_t])
                        //       h_t = o_t * tanh(c_t)
                        timestep <= timestep + 1;
                    end else begin
                        if (layer_idx < NUM_LAYERS - 1) begin
                            layer_idx <= layer_idx + 1;
                            timestep  <= '0;
                        end else begin
                            lstm_state <= LSTM_BACKWARD;
                            timestep   <= TIME_STEPS - 1;
                            layer_idx  <= '0;
                        end
                    end
                end
                
                LSTM_BACKWARD: begin
                    // Process backward (right-to-left)
                    if (timestep > 0) begin
                        timestep <= timestep - 1;
                    end else begin
                        if (layer_idx < NUM_LAYERS - 1) begin
                            layer_idx <= layer_idx + 1;
                            timestep  <= TIME_STEPS - 1;
                        end else begin
                            lstm_state <= LSTM_COMBINE;
                            timestep   <= '0;
                        end
                    end
                end
                
                LSTM_COMBINE: begin
                    // Concatenate forward and backward hidden states
                    if (timestep < TIME_STEPS) begin
                        for (int i = 0; i < HIDDEN_SIZE; i++) begin
                            // Simple average (real: concatenate + linear layer)
                            o_output[timestep][i] <= 
                                (h_forward[NUM_LAYERS-1][timestep][i] + 
                                 h_backward[NUM_LAYERS-1][timestep][i]) >> 1;
                        end
                        timestep <= timestep + 1;
                    end else begin
                        lstm_state <= LSTM_SOFTMAX;
                        timestep   <= '0;
                    end
                end
                
                LSTM_SOFTMAX: begin
                    // Apply softmax to get probabilities
                    // (Simplified: just normalize, real softmax uses exp)
                    if (timestep < TIME_STEPS) begin
                        timestep <= timestep + 1;
                    end else begin
                        lstm_state <= LSTM_DONE;
                    end
                end
                
                LSTM_DONE: begin
                    o_done     <= 1'b1;
                    lstm_state <= LSTM_IDLE;
                end
            endcase
        end
    end

endmodule

module hw_decoder_crf_beamsearch #(
    parameter TIME_STEPS    = 1667,
    parameter FEATURE_DIM   = 384,
    parameter BEAM_WIDTH    = 5,
    parameter MAX_SEQ_LEN   = 512
) (
    input wire clk,
    input wire reset_n,
    
    // Input from Stage 2
    input wire [127:0] i_read_id,
    input wire [31:0]  i_feature_data,
    input wire         i_is_last_chunk,
    input wire         i_end_of_chunk,
    input wire         i_feature_valid,
    output reg         o_feature_ready,
    
    // Output to Stage 4
    output reg [127:0] o_read_id,
    output reg [7:0]   o_base_char,
    output reg [7:0]   o_qual_char,
    output reg         o_end_of_read,
    output reg         o_base_valid,
    input  wire        i_base_ready
);

    //==================================================================
    // State Machine
    //==================================================================
    localparam IDLE           = 3'd0;
    localparam BUFFER_FEATURE = 3'd1;
    localparam DECODE         = 3'd2;
    localparam OUTPUT_BASES   = 3'd3;
    
    reg [2:0] state;
    
    //==================================================================
    // Feature Buffer - stores incoming probabilities
    //==================================================================
    reg [31:0] features [0:TIME_STEPS-1][0:4];  // 5 classes: A/C/G/T/Blank
    reg [15:0] feature_time;
    reg [2:0]  feature_class;
    
    //==================================================================
    // Output Buffer
    //==================================================================
    reg [7:0]  output_seq [0:MAX_SEQ_LEN-1];
    reg [7:0]  output_qual [0:MAX_SEQ_LEN-1];
    reg [15:0] output_len;
    reg [15:0] output_idx;
    
    //==================================================================
    // Working registers
    //==================================================================
    reg [127:0] stored_read_id;
    reg [15:0]  decode_time;
    reg [2:0]   last_base;  // Track last emitted base
    
    //==================================================================
    // Temporary variables for decode stage
    //==================================================================
    reg [31:0] max_prob;
    reg [2:0]  max_base;
    integer i;
    
    //==================================================================
    // Helper: Convert base index to ASCII
    //==================================================================
    function [7:0] base_to_ascii;
        input [2:0] base;
        begin
            case (base)
                3'd0: base_to_ascii = 8'd65;  // 'A'
                3'd1: base_to_ascii = 8'd67;  // 'C'
                3'd2: base_to_ascii = 8'd71;  // 'G'
                3'd3: base_to_ascii = 8'd84;  // 'T'
                default: base_to_ascii = 8'd78;  // 'N'
            endcase
        end
    endfunction
    
    //==================================================================
    // Main State Machine
    //==================================================================
    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            state <= IDLE;
            o_feature_ready <= 1'b0;
            o_base_valid <= 1'b0;
            o_end_of_read <= 1'b0;
            feature_time <= 16'd0;
            feature_class <= 3'd0;
            output_idx <= 16'd0;
            output_len <= 16'd0;
            last_base <= 3'd4;  // Start with blank
            decode_time <= 16'd0;
            stored_read_id <= 128'd0;
            
        end else begin
            case (state)
                //==============================================
                IDLE: begin
                    o_feature_ready <= 1'b1;
                    o_base_valid <= 1'b0;
                    o_end_of_read <= 1'b0;
                    feature_time <= 16'd0;
                    feature_class <= 3'd0;
                    output_idx <= 16'd0;
                    output_len <= 16'd0;
                    last_base <= 3'd4;
                    decode_time <= 16'd0;
                    
                    if (i_feature_valid) begin
                        stored_read_id <= i_read_id;
                        state <= BUFFER_FEATURE;
                    end
                end
                
                //==============================================
                BUFFER_FEATURE: begin
                    if (i_feature_valid && o_feature_ready) begin
                        // Store feature data
                        features[feature_time][feature_class] <= i_feature_data;
                        
                        // Advance to next class
                        if (feature_class == 3'd4) begin
                            feature_class <= 3'd0;
                            
                            // Check if we're done with all timesteps
                            if (feature_time == TIME_STEPS - 1) begin
                                state <= DECODE;
                                decode_time <= 16'd0;
                                o_feature_ready <= 1'b0;
                            end else begin
                                feature_time <= feature_time + 16'd1;
                            end
                        end else begin
                            feature_class <= feature_class + 3'd1;
                        end
                    end
                end
                
                //==============================================
                DECODE: begin
                    if (decode_time < TIME_STEPS) begin
                        // Find max probability base for this timestep
                        max_prob = features[decode_time][0];
                        max_base = 3'd0;
                        
                        // Compare all 5 classes
                        for (i = 1; i < 5; i = i + 1) begin
                            if (features[decode_time][i] > max_prob) begin
                                max_prob = features[decode_time][i];
                                max_base = i[2:0];
                            end
                        end
                        
                        // CTC rules: emit base if not blank and different from previous
                        if (max_base != 3'd4 && max_base != last_base) begin
                            output_seq[output_len] <= base_to_ascii(max_base);
                            output_qual[output_len] <= 8'd63;  // Phred+33 = '?' (Q30)
                            output_len <= output_len + 16'd1;
                        end
                        
                        last_base <= max_base;
                        decode_time <= decode_time + 16'd1;
                        
                    end else begin
                        // Decoding complete
                        state <= OUTPUT_BASES;
                    end
                end
                
                //==============================================
                OUTPUT_BASES: begin
                    o_base_valid <= 1'b1;
                    o_read_id <= stored_read_id;
                    o_base_char <= output_seq[output_idx];
                    o_qual_char <= output_qual[output_idx];
                    
                    // Mark last base
                    if (output_idx == output_len - 1) begin
                        o_end_of_read <= 1'b1;
                    end else begin
                        o_end_of_read <= 1'b0;
                    end
                    
                    if (i_base_ready) begin
                        if (output_idx >= output_len - 1) begin
                            state <= IDLE;
                            o_base_valid <= 1'b0;
                            o_end_of_read <= 1'b0;
                        end else begin
                            output_idx <= output_idx + 16'd1;
                        end
                    end
                end
                
                //==============================================
                default: state <= IDLE;
            endcase
        end
    end

endmodule

//======================================================================
// STAGE 4: Chunk Stitcher & FASTQ Formatter
//======================================================================
module fastq_stitcher_formatter (
    input wire clk,
    input wire reset_n,
    
    input wire [127:0] i_read_id,
    input wire [7:0]   i_base_char,
    input wire [7:0]   i_qual_char,
    input wire         i_end_of_read,
    input wire         i_base_valid,
    output wire        o_base_ready,
    
    output wire [7:0]  o_fastq_char,
    output wire        o_end_of_read,
    output wire        o_fastq_valid,
    input  wire        i_fastq_ready
);

    // Simple pass-through
    assign o_base_ready  = i_fastq_ready;
    assign o_fastq_valid = i_base_valid;
    assign o_fastq_char  = i_base_char;
    assign o_end_of_read = i_end_of_read;

endmodule

//======================================================================
// STAGE 5: FASTQ Quality Control & Trimming
//======================================================================
module fastq_qc_trimmer #(
    parameter MAX_READ_LENGTH = 150
) (
    input wire clk,
    input wire reset_n,
    input wire start,
    
    input wire [7:0]   i_fastq_char_in,
    input wire         i_end_of_read_in,
    input wire         i_fastq_valid_in,
    output wire        o_fastq_ready_out,
    
    output wire [7:0]   o_fastq_char,
    output wire [127:0] o_read_id,
    output wire         o_end_of_read,
    output wire         o_fastq_valid,
    input  wire         i_fastq_ready
);

    // Pass-through with dummy read_id
    assign o_fastq_ready_out = i_fastq_ready;
    assign o_fastq_valid     = i_fastq_valid_in;
    assign o_fastq_char      = i_fastq_char_in;
    assign o_end_of_read     = i_end_of_read_in;
    assign o_read_id         = 128'h0123456789ABCDEF0123456789ABCDEF;

endmodule

//======================================================================
// STAGE 6A: Transcriptome Aligner (Host)
//======================================================================
module transcriptome_aligner_host #(
    parameter MAX_READ_LENGTH = 150,
    parameter NUM_HOST_GENES  = 20000
) (
    input wire clk,
    input wire reset_n,
    
    input wire [7:0]   i_fastq_char,
    input wire [127:0] i_read_id,
    input wire         i_end_of_read,
    input wire         i_fastq_valid,
    output reg         o_fastq_ready,
    
    output reg [127:0] o_read_id,
    output reg [31:0]  o_gene_id,
    output reg [7:0]   o_mapping_quality,
    output reg         o_is_host_aligned,
    output reg         o_align_valid,
    input  wire        i_align_ready
);

    reg [1:0] state;
    localparam IDLE = 2'd0;
    localparam RECV = 2'd1;
    localparam SEND = 2'd2;

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            state <= IDLE;
            o_align_valid <= 1'b0;
            o_fastq_ready <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    o_align_valid <= 1'b0;
                    o_fastq_ready <= 1'b1;
                    if (i_fastq_valid) begin
                        state <= RECV;
                        o_read_id <= i_read_id;
                    end
                end
                
                RECV: begin
                    o_fastq_ready <= 1'b1;
                    if (i_fastq_valid && i_end_of_read) begin
                        state <= SEND;
                        o_fastq_ready <= 1'b0;
                        o_gene_id <= 32'd1000;
                        o_mapping_quality <= 8'd60;
                        o_is_host_aligned <= 1'b1;
                    end
                end
                
                SEND: begin
                    o_align_valid <= 1'b1;
                    if (i_align_ready) begin
                        state <= IDLE;
                        o_align_valid <= 1'b0;
                    end
                end
            endcase
        end
    end

endmodule

//======================================================================
// STAGE 6B: Pathogen Aligner (Parallel)
//======================================================================
module pathogen_aligner #(
    parameter MAX_READ_LENGTH = 150,
    parameter NUM_PATHOGENS   = 1000
) (
    input wire clk,
    input wire reset_n,
    
    input wire [7:0]   i_fastq_char,
    input wire [127:0] i_read_id,
    input wire         i_end_of_read,
    input wire         i_fastq_valid,
    output reg         o_fastq_ready,
    
    output reg [127:0] o_read_id,
    output reg [15:0]  o_pathogen_id,
    output reg [7:0]   o_mapping_quality,
    output reg         o_is_pathogen,
    output reg         o_align_valid,
    input  wire        i_align_ready
);

    reg [1:0] state;
    localparam IDLE = 2'd0;
    localparam RECV = 2'd1;
    localparam SEND = 2'd2;

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            state <= IDLE;
            o_align_valid <= 1'b0;
            o_fastq_ready <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    o_align_valid <= 1'b0;
                    o_fastq_ready <= 1'b1;
                    if (i_fastq_valid) begin
                        state <= RECV;
                        o_read_id <= i_read_id;
                    end
                end
                
                RECV: begin
                    o_fastq_ready <= 1'b1;
                    if (i_fastq_valid && i_end_of_read) begin
                        state <= SEND;
                        o_fastq_ready <= 1'b0;
                        o_pathogen_id <= 16'd42;
                        o_mapping_quality <= 8'd55;
                        o_is_pathogen <= 1'b1;
                    end
                end
                
                SEND: begin
                    o_align_valid <= 1'b1;
                    if (i_align_ready) begin
                        state <= IDLE;
                        o_align_valid <= 1'b0;
                    end
                end
            endcase
        end
    end

endmodule

//======================================================================
// STAGE 7: Gene Expression Quantifier
//======================================================================
module gene_expression_quantifier #(
    parameter NUM_HOST_GENES = 20000
) (
    input wire clk,
    input wire reset_n,
    
    input wire [127:0] i_read_id,
    input wire [31:0]  i_gene_id,
    input wire [7:0]   i_mapping_quality,
    input wire         i_is_host_aligned,
    input wire         i_align_valid,
    output wire        o_align_ready,
    
    output wire [31:0] o_gene_id,
    output wire [31:0] o_raw_count,
    output wire [31:0] o_tpm,
    output wire        o_count_valid,
    input  wire        i_count_ready
);

    // Pass-through with dummy quantification
    assign o_align_ready = i_count_ready;
    assign o_count_valid = i_align_valid;
    assign o_gene_id     = i_gene_id;
    assign o_raw_count   = 32'd100;
    assign o_tpm         = 32'd250;

endmodule

//======================================================================
// STAGE 8: Pathogen Quantifier
//======================================================================
module pathogen_quantifier #(
    parameter NUM_PATHOGENS = 1000
) (
    input wire clk,
    input wire reset_n,
    
    input wire [127:0] i_read_id,
    input wire [15:0]  i_pathogen_id,
    input wire [7:0]   i_mapping_quality,
    input wire         i_is_pathogen,
    input wire         i_align_valid,
    output wire        o_align_ready,
    
    output wire [15:0] o_pathogen_id,
    output wire [31:0] o_pathogen_count,
    output wire [7:0]  o_pathogen_abundance,
    output wire        o_pathogen_valid,
    input  wire        i_pathogen_ready
);

    // Pass-through with dummy quantification
    assign o_align_ready       = i_pathogen_ready;
    assign o_pathogen_valid    = i_align_valid;
    assign o_pathogen_id       = i_pathogen_id;
    assign o_pathogen_count    = 32'd50;
    assign o_pathogen_abundance = 8'd75;

endmodule

//======================================================================
// STAGE 9: Biomarker Panel Analyzer
//======================================================================
module biomarker_analyzer #(
    parameter NUM_BIOMARKERS = 50
) (
    input wire clk,
    input wire reset_n,
    
    input wire [31:0] i_gene_id,
    input wire [31:0] i_raw_count,
    input wire [31:0] i_tpm,
    input wire        i_count_valid,
    output wire       o_count_ready,
    
    output wire [7:0]  o_biomarker_id,
    output wire [31:0] o_expression_level,
    output wire [7:0]  o_fold_change,
    output wire        o_biomarker_valid,
    input  wire        i_biomarker_ready
);

    // Pass-through with biomarker data
    assign o_count_ready      = i_biomarker_ready;
    assign o_biomarker_valid  = i_count_valid;
    assign o_biomarker_id     = i_gene_id[7:0];
    assign o_expression_level = i_tpm;
    assign o_fold_change      = 8'd3;

endmodule

//======================================================================
// STAGE 10: ML Sepsis Classifier
//======================================================================
module sepsis_ml_classifier #(
    parameter NUM_BIOMARKERS = 50
) (
    input wire clk,
    input wire reset_n,
    
    input wire [7:0]   i_biomarker_id,
    input wire [31:0]  i_expression_level,
    input wire [7:0]   i_fold_change,
    input wire         i_biomarker_valid,
    output reg         o_biomarker_ready,
    
    input wire [15:0]  i_pathogen_id,
    input wire [31:0]  i_pathogen_count,
    input wire [7:0]   i_pathogen_abundance,
    input wire         i_pathogen_valid,
    output reg         o_pathogen_ready,
    
    output reg         o_sepsis_detected,
    output reg [7:0]   o_confidence,
    output reg [7:0]   o_severity_score,
    output reg [15:0]  o_top_pathogen_id,
    output reg [7:0]   o_pathogen_load,
    output reg         o_classification_valid,
    input  wire        i_classification_ready
);

    reg [1:0] state;
    localparam IDLE = 2'd0;
    localparam WAIT = 2'd1;
    localparam SEND = 2'd2;
    
    reg got_biomarker;
    reg got_pathogen;

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            state <= IDLE;
            o_classification_valid <= 1'b0;
            o_biomarker_ready <= 1'b0;
            o_pathogen_ready <= 1'b0;
            got_biomarker <= 1'b0;
            got_pathogen <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    o_classification_valid <= 1'b0;
                    o_biomarker_ready <= 1'b1;
                    o_pathogen_ready <= 1'b1;
                    got_biomarker <= 1'b0;
                    got_pathogen <= 1'b0;
                    
                    if (i_biomarker_valid || i_pathogen_valid) begin
                        state <= WAIT;
                        if (i_biomarker_valid) got_biomarker <= 1'b1;
                        if (i_pathogen_valid) got_pathogen <= 1'b1;
                    end
                end
                
                WAIT: begin
                    if (i_biomarker_valid && !got_biomarker) begin
                        got_biomarker <= 1'b1;
                    end
                    if (i_pathogen_valid && !got_pathogen) begin
                        got_pathogen <= 1'b1;
                    end
                    
                    if (got_biomarker && got_pathogen) begin
                        state <= SEND;
                        o_biomarker_ready <= 1'b0;
                        o_pathogen_ready <= 1'b0;
                        o_sepsis_detected <= 1'b1;
                        o_confidence <= 8'd85;
                        o_severity_score <= 8'd65;
                        o_top_pathogen_id <= i_pathogen_id;
                        o_pathogen_load <= i_pathogen_abundance;
                    end
                end
                
                SEND: begin
                    o_classification_valid <= 1'b1;
                    if (i_classification_ready) begin
                        state <= IDLE;
                        o_classification_valid <= 1'b0;
                    end
                end
            endcase
        end
    end

endmodule

//======================================================================
// STAGE 11: Clinical Report Generator
//======================================================================
module clinical_report_formatter (
    input wire clk,
    input wire reset_n,
    
    input wire         i_sepsis_detected,
    input wire [7:0]   i_confidence,
    input wire [7:0]   i_severity_score,
    input wire [15:0]  i_top_pathogen_id,
    input wire [7:0]   i_pathogen_load,
    input wire         i_classification_valid,
    output reg         o_classification_ready,
    
    output reg [7:0]   o_report_char,
    output reg         o_report_valid,
    input  wire        i_report_ready
);

    reg [2:0] state;
    localparam IDLE = 3'd0;
    localparam SEND = 3'd1;
    
    reg [4:0] char_idx;
    
    // Simple report message: "SEPSIS DETECTED\n"
    reg [7:0] report_msg [0:15];
    
    initial begin
        report_msg[0]  = "S";
        report_msg[1]  = "E";
        report_msg[2]  = "P";
        report_msg[3]  = "S";
        report_msg[4]  = "I";
        report_msg[5]  = "S";
        report_msg[6]  = " ";
        report_msg[7]  = "D";
        report_msg[8]  = "E";
        report_msg[9]  = "T";
        report_msg[10] = "E";
        report_msg[11] = "C";
        report_msg[12] = "T";
        report_msg[13] = "E";
        report_msg[14] = "D";
        report_msg[15] = "\n";
    end

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            state <= IDLE;
            o_report_valid <= 1'b0;
            o_classification_ready <= 1'b0;
            char_idx <= 5'd0;
        end else begin
            case (state)
                IDLE: begin
                    o_report_valid <= 1'b0;
                    o_classification_ready <= 1'b1;
                    char_idx <= 5'd0;
                    
                    if (i_classification_valid) begin
                        state <= SEND;
                        o_classification_ready <= 1'b0;
                    end
                end
                
                SEND: begin
                    o_report_valid <= 1'b1;
                    o_report_char <= report_msg[char_idx];
                    
                    if (i_report_ready) begin
                        if (char_idx == 5'd15) begin
                            state <= IDLE;
                            o_report_valid <= 1'b0;
                        end else begin
                            char_idx <= char_idx + 5'd1;
                        end
                    end
                end
            endcase
        end
    end

endmodule
