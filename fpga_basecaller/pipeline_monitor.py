import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import serial
import serial.tools.list_ports
import threading
import queue
import time
import math
import random # Added for mock data

# --- Xgome App Configuration ---
DEFAULT_BAUD_RATE = 115200
APP_NAME = "Xgome"
APP_TITLE = "Xgome Pipeline Monitor"
MOCK_MODE = True # <<< NEW: Set to False to try real serial port

# --- Color Palette (Xgome Industrial Metallic) ---
COLOR_BG = "#2B2B2B"              # Deep charcoal
COLOR_BG_LIGHT = "#3C3C3C"        # Lighter charcoal (panel borders)
COLOR_BG_PANEL = "#222222"        # Panel inset background
COLOR_FG = "#E0E0E0"              # Light grey text
COLOR_ACCENT = "#00AEEF"          # Xgome Tech Blue (glowing)
COLOR_ACCENT_DARK = "#007A9E"     # Accent hover/click
COLOR_SUCCESS = "#00D09C"          # Bright mint green
COLOR_ERROR = "#FF5B5B"            # Bright red
COLOR_TEXT_BOX = "#1C1C1C"        # Very dark (for gene visualizer bg)

# --- Gene Sphere Colors ---
BASE_COLORS = {
    'A': '#FF5B5B', # Red
    'C': '#00AEEF', # Blue
    'G': '#00D09C', # Green
    'T': '#F0A830', # Orange
    'U': '#A85BFF', # Purple
    'N': '#666666'  # Grey for unknown
}

# -------------------------------------------------------------------
#  MOCK Serial Device for Visualization
# -------------------------------------------------------------------
if MOCK_MODE:
    class MockSerial:
        """Simulates a serial port device for testing the GUI."""
        def __init__(self, port, baud, timeout=1):
            # The simulator will cycle through data for visualization
            self.mock_data = self._generate_mock_sequence()
            self.data_index = 0
            self.read_delay = 0.05 # How fast data is 'received'
            self.is_open = False
            self.in_waiting = 0
            self.port = port
            self.baud = baud
        
        def _generate_mock_sequence(self):
            """Generates a list of simulated serial messages."""
            messages = [
                'S,Read_12345678', # Start
            ]
            
            # --- Progress Simulation ---
            # Basecalling 0-100%
            messages.extend([f'B,{i}' for i in range(0, 101, 10)])
            # Alignment 0-100%
            messages.extend([f'A,{i}' for i in range(0, 101, 10)])
            # Variant Calling 0-100%
            messages.extend([f'V,{i}' for i in range(0, 101, 10)])

            # --- Gene Data Simulation ---
            # A short, complex gene sequence
            gene_sequence = "GATTACAATCGATCGATCGTCAGTCCAGTCGTAGCTAGCATGCCGATCGATCGATCGATCGTACGTAGCATG"
            chunk_size = 5
            for i in range(0, len(gene_sequence), chunk_size):
                chunk = gene_sequence[i:i+chunk_size]
                messages.append(f'D,{chunk}') # Data chunk
                # Add random progress updates to simulate background processing
                if random.random() < 0.2:
                    messages.append(f'V,{random.randint(50, 95)}') 
                    messages.append(f'A,{random.randint(50, 95)}')
            
            messages.append('F,Success') # Finished
            return messages

        def open(self):
            self.is_open = True
            self.data_index = 0
            self.in_waiting = 1 # Start with data waiting
        
        def close(self):
            self.is_open = False
            
        def readline(self):
            """Simulates reading a line of data."""
            if self.data_index < len(self.mock_data):
                line = self.mock_data[self.data_index]
                self.data_index += 1
                self.in_waiting = 1 # More data is 'available' after reading
                time.sleep(self.read_delay)
                return (line + '\r\n').encode('ascii')
            else:
                # Loop the sequence for continuous visualization
                self.data_index = 0
                return b'' # Empty byte string when no data is ready for now

# -------------------------------------------------------------------
#  Serial Handler (Runs in a separate thread)
# -------------------------------------------------------------------
class SerialHandler:
    """
    Manages the serial connection and I/O in a dedicated thread.
    Communicates with the main GUI thread via queues.
    """
    def __init__(self, in_queue, out_queue):
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.serial_port = None
        self.running = threading.Event()
        self.running.set()

    def run(self):
        """Main loop for the serial thread."""
        port = None
        baud = None
        
        # Determine which Serial class to use
        SerialClass = MockSerial if MOCK_MODE else serial.Serial

        while self.running.is_set():
            try:
                # Check for commands from the GUI thread
                cmd, payload = self.in_queue.get(timeout=0.01)
                if cmd == 'connect':
                    if self.serial_port and self.serial_port.is_open:
                        self.serial_port.close()
                    port, baud = payload
                    self.serial_port = None
                    self.out_queue.put(('status', f'Connecting to {port}...', 'normal'))
                elif cmd == 'stop':
                    self.running.clear()
                    break
            except queue.Empty:
                pass # No command from GUI, continue loop
                
            if not port:
                time.sleep(0.1)
                continue
            
            # --- Connection Logic ---
            if self.serial_port is None:
                try:
                    # Use the appropriate SerialClass (real or mock)
                    self.serial_port = SerialClass(port, baud, timeout=1) 
                    self.serial_port.open() # MockSerial needs explicit open
                    self.out_queue.put(('status', f'Connected to {port} at {baud} bps', 'success'))
                except serial.SerialException as e:
                    # Only attempt to reconnect if in real mode
                    if not MOCK_MODE:
                        self.out_queue.put(('status', f'Connection failed: {e}. Retrying...', 'error'))
                        time.sleep(3)
                    else:
                        # In mock mode, if connection fails, just stop the loop
                        self.out_queue.put(('status', 'Mock connection failed (Check MOCK_MODE setting).', 'error'))
                        self.running.clear()
                        break 
                except Exception as e:
                    self.out_queue.put(('status', f'Serial thread error during connect: {e}', 'error'))
                    if not MOCK_MODE:
                        time.sleep(3)
                    else:
                        self.running.clear()
                        break 
                continue # Skip to next loop iteration after connection attempt
            
            # --- Data Reading Logic ---
            try:
                # Use in_waiting for efficiency, but also as a signal from MockSerial
                if self.serial_port.in_waiting > 0: 
                    line = self.serial_port.readline()
                    
                    if not line: # MockSerial returns b'' when sequence is done
                        self.serial_port.in_waiting = 0 
                        time.sleep(0.1)
                        continue
                        
                    try:
                        decoded_line = line.decode('ascii').strip()
                        if decoded_line:
                            self.out_queue.put(('data', decoded_line))
                    except UnicodeDecodeError:
                        self.out_queue.put(('log', f'Received non-ASCII data: {line}'))
                        
            except serial.SerialException as e:
                self.out_queue.put(('status', f'Connection lost: {e}. Reconnecting...', 'error'))
                if self.serial_port:
                    self.serial_port.close()
                self.serial_port = None
                time.sleep(3)
            except Exception as e:
                self.out_queue.put(('status', f'Serial thread error during read: {e}', 'error'))
                time.sleep(0.01) # Small sleep to prevent tight loop on error

        # --- Cleanup ---
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        self.out_queue.put(('status', 'Disconnected.', 'normal'))
        print("Serial handler thread stopped.")

# -------------------------------------------------------------------
#  Tooltip Helper Class (No changes needed)
# -------------------------------------------------------------------
class Tooltip:
    """Simple and professional tooltip for any widget."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        try:
            x, y, _, _ = self.widget.bbox("insert")
            x += self.widget.winfo_rootx() + 25
            y += self.widget.winfo_rooty() + 20
        except tk.TclError: # Fallback for widgets without "insert" bbox
            x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
            
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip_window, text=self.text,
                         background=COLOR_BG_LIGHT, foreground=COLOR_FG,
                         relief=tk.SOLID, borderwidth=1,
                         font=("Helvetica", 9))
        label.pack()

    def hide_tip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None

# -------------------------------------------------------------------
#  Scientific Gene Visualizer Widget (No changes needed)
# -------------------------------------------------------------------
class GeneVisualizer(tk.Canvas):
    """
    A custom canvas widget that draws a sequence of bases as
    "metallic spheres" and handles wrapping and auto-scrolling.
    """
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        
        self.sequence = ""
        # Initialize with a non-zero width to prevent division by zero errors
        self.current_width = 1 
        
        # --- Sphere drawing parameters ---
        self.sphere_diam = 18
        self.sphere_padding_x = 4
        self.sphere_padding_y = 6
        self.margin = 10
        self.sphere_font = ("Consolas", 9, "bold")
        
        # Specular highlight (for "metallic" look)
        self.highlight_offset = 3
        self.highlight_diam = 4
        
        # Bind resize event to re-wrap the sequence
        self.bind("<Configure>", self._on_resize)
        
    def _on_resize(self, event):
        """Redraw sequence when widget width changes."""
        # Update width, ensuring it's at least 1
        new_width = max(1, event.width)
        if new_width != self.current_width:
            self.current_width = new_width
            self._redraw_sequence()

    def add_bases(self, bases):
        """Public method to add new bases to the sequence."""
        self.sequence += bases
        self._redraw_sequence()
        # Auto-scroll to the end
        self.yview_moveto(1.0)

    def clear(self):
        """Clears the canvas and resets the sequence."""
        self.sequence = ""
        self.delete("all")
        
    def _redraw_sequence(self):
        """Internal method to draw the entire sequence."""
        self.delete("all")
        if self.current_width <= (2 * self.margin): # Not wide enough
            return

        x = self.margin
        y = self.margin
        total_sphere_width = self.sphere_diam + self.sphere_padding_x
        
        for base in self.sequence:
            # --- Wrap logic ---
            if x + self.sphere_diam > self.current_width - self.margin:
                x = self.margin
                y += self.sphere_diam + self.sphere_padding_y
            
            # --- Draw "metallic" sphere ---
            color = BASE_COLORS.get(base.upper(), BASE_COLORS['N'])
            
            # 1. Main sphere color
            self.create_oval(
                x, y, 
                x + self.sphere_diam, y + self.sphere_diam,
                fill=color, outline=""
            )
            
            # 2. Specular highlight (the "metallic" part)
            self.create_oval(
                x + self.highlight_offset, y + self.highlight_offset,
                x + self.highlight_offset + self.highlight_diam, 
                y + self.highlight_offset + self.highlight_diam,
                fill="#FFFFFF", outline=""
            )
            
            # 3. Base letter
            self.create_text(
                x + self.sphere_diam / 2, y + self.sphere_diam / 2,
                text=base.upper(), fill=COLOR_TEXT_BOX, font=self.sphere_font
            )
            
            # Increment x position
            x += total_sphere_width

        # Update the canvas scroll region
        self.config(scrollregion=self.bbox("all"))

# -------------------------------------------------------------------
#  Main GUI Application
# -------------------------------------------------------------------
class PipelineMonitor(tk.Tk):
    
    def __init__(self):
        super().__init__()
        
        self.title(APP_TITLE)
        self.geometry("850x700") # Made taller for visualizer
        
        # Threading and queues
        self.serial_to_gui_queue = queue.Queue()
        self.gui_to_serial_queue = queue.Queue()
        self.serial_handler_thread = None
        self.is_connected = False
        
        self.progress_bars = {}
        self.progress_labels = {}
        
        self.setup_style()
        self.create_widgets()
        
        self.start_serial_handler()
        self.poll_queue()
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_style(self):
        """Configures the 'xgome' custom ttk theme."""
        self.style = ttk.Style()
        self.style.theme_use('clam') 

        # --- Configure 'xgome' theme ---
        self.style.configure('.',
                             background=COLOR_BG,
                             foreground=COLOR_FG,
                             fieldbackground=COLOR_BG_LIGHT,
                             font=('Helvetica', 10))

        # --- Specific Widgets ---
        self.style.configure('TFrame', background=COLOR_BG)
        self.style.configure('TLabel', background=COLOR_BG, foreground=COLOR_FG)
        
        self.style.configure('Title.TLabel',
                             font=('Helvetica', 18, 'bold'),
                             foreground=COLOR_ACCENT)
                             
        self.style.configure('Logo.TLabel',
                             font=('Arial', 28, 'bold'), # Bigger logo
                             foreground=COLOR_ACCENT)

        # *** "Metallic" Panel Style FIX ***
        # This is the robust, cross-platform way to fix the layout errors.
        # We try all common layout names until one works.
        try:
            base_layout = self.style.layout('TLabelFrame') 
        except tk.TclError:
            try:
                base_layout = self.style.layout('Labelframe')
            except tk.TclError:
                try:
                    base_layout = self.style.layout('TLabelframe')
                except tk.TclError:
                    print("Warning: Could not find base labelframe layout. Using default.")
                    base_layout = None

        if base_layout:
            self.style.layout('Metallic.TLabelFrame', base_layout)
        
        self.style.configure('Metallic.TLabelFrame',
                             background=COLOR_BG_PANEL, # Inset panel
                             bordercolor=COLOR_BG_LIGHT,
                             borderwidth=2,
                             relief=tk.RIDGE) # Gives a "bolted on" look
        
        # *** FIX 1: Correct sub-element name ***
        # The sub-element for the title is '.Label' (capital L), not '.label'
        self.style.configure('Metallic.TLabelFrame.Label', 
                             background=COLOR_BG_PANEL,
                             foreground=COLOR_FG,
                             font=('Helvetica', 12, 'bold'))
        
        # *** FIX 2: Create a NEW style for internal labels ***
        # This is the style for regular ttk.Labels that are *inside* the panel.
        self.style.configure('Panel.TLabel',
                             background=COLOR_BG_PANEL,
                             foreground=COLOR_FG,
                             font=('Helvetica', 10, 'bold')) # Match font
        # --- END OF FIX ---


        self.style.configure('TButton',
                             background=COLOR_ACCENT,
                             foreground=COLOR_TEXT_BOX,
                             font=('Helvetica', 10, 'bold'),
                             bordercolor=COLOR_ACCENT,
                             relief=tk.FLAT,
                             padding=(10, 5))
        self.style.map('TButton',
                       background=[('active', COLOR_ACCENT_DARK), ('disabled', COLOR_BG_LIGHT)],
                       foreground=[('disabled', COLOR_FG)])

        self.style.configure('Red.TButton',
                             background=COLOR_ERROR,
                             bordercolor=COLOR_ERROR)
        self.style.map('Red.TButton', background=[('active', '#C04040')])
        
        self.style.configure('TCombobox',
                             arrowcolor=COLOR_ACCENT,
                             fieldbackground=COLOR_BG_LIGHT,
                             background=COLOR_BG_LIGHT,
                             selectbackground=COLOR_ACCENT,
                             selectforeground=COLOR_TEXT_BOX)
        self.option_add('*TCombobox*Listbox.background', COLOR_BG_LIGHT)
        self.option_add('*TCombobox*Listbox.foreground', COLOR_FG)
        self.option_add('*TCombobox*Listbox.selectBackground', COLOR_ACCENT)

        self.style.configure('Horizontal.TProgressbar',
                             troughcolor=COLOR_BG_LIGHT,
                             background=COLOR_ACCENT,
                             bordercolor=COLOR_BG_LIGHT)
                             
        self.style.configure('Success.Horizontal.TProgressbar', background=COLOR_SUCCESS)
        
        self.style.configure('Vertical.TScrollbar',
                             background=COLOR_BG_PANEL,
                             troughcolor=COLOR_BG_LIGHT,
                             bordercolor=COLOR_BG_PANEL,
                             arrowcolor=COLOR_ACCENT,
                             relief=tk.FLAT)
        self.style.map('Vertical.TScrollbar',
                       background=[('active', COLOR_ACCENT_DARK)])

        self.style.configure('Status.TLabel', padding=5)
        self.style.configure('Status.Normal.TLabel', background=COLOR_BG_LIGHT, foreground=COLOR_FG)
        self.style.configure('Status.Success.TLabel', background=COLOR_SUCCESS, foreground=COLOR_TEXT_BOX)
        self.style.configure('Status.Error.TLabel', background=COLOR_ERROR, foreground=COLOR_FG)

    def create_widgets(self):
        """Builds all the GUI elements."""
        
        self.configure(background=COLOR_BG)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1) # Give row 3 (visualizer) expansion
        
        # --- 1. Title Bar ---
        title_frame = ttk.Frame(self, padding=(10, 10))
        title_frame.grid(row=0, column=0, sticky='ew')
        
        logo_label = ttk.Label(title_frame, text=f"[{APP_NAME}]", style='Logo.TLabel')
        logo_label.pack(side=tk.LEFT, padx=(0, 15))
        
        title_label = ttk.Label(title_frame, text=APP_TITLE, style='Title.TLabel')
        title_label.pack(side=tk.LEFT, anchor='w')
        
        # --- 2. Connection Bar ---
        # This now correctly uses the 'Metallic.TLabelFrame' style
        conn_frame = ttk.LabelFrame(self, text="FPGA Connection",
                                    style='Metallic.TLabelFrame', padding=(10, 10))
        conn_frame.grid(row=1, column=0, sticky='ew', padx=10, pady=5)
        
        # Use a list of real ports or just 'MOCK_PORT' in mock mode
        if MOCK_MODE:
            available_ports = ["MOCK_PORT (Simulator Active)"]
        else:
            available_ports = [port.device for port in serial.tools.list_ports.comports()]
        
        # *** STYLE FIX: Use the new 'Panel.TLabel' style ***
        ttk.Label(conn_frame, text="Port:", style='Panel.TLabel').pack(side=tk.LEFT, padx=(0, 5))
        self.port_var = tk.StringVar(value=available_ports[0] if available_ports else "")
        self.port_combo = ttk.Combobox(conn_frame, textvariable=self.port_var, width=25, 
                                        values=available_ports, state='readonly') # 'readonly' prevents typing
        self.port_combo.pack(side=tk.LEFT, padx=5)
        Tooltip(self.port_combo, "Select the serial port for the FPGA.")
        
        # *** STYLE FIX: Use the new 'Panel.TLabel' style ***
        ttk.Label(conn_frame, text="Baud:", style='Panel.TLabel').pack(side=tk.LEFT, padx=(10, 5))
        self.baud_var = tk.StringVar(value=str(DEFAULT_BAUD_RATE))
        self.baud_entry = ttk.Entry(conn_frame, textvariable=self.baud_var, width=10)
        self.baud_entry.pack(side=tk.LEFT, padx=5)
        Tooltip(self.baud_entry, "Set the baud rate (e.g., 115200).")

        self.connect_button = ttk.Button(conn_frame, text="Connect",
                                         command=self.toggle_connection, width=15)
        self.connect_button.pack(side=tk.RIGHT, padx=5)
        
        # --- 3. Pipeline Stages ---
        stages_frame = ttk.LabelFrame(self, text="Pipeline Status",
                                      style='Metallic.TLabelFrame', padding=(10, 10))
        stages_frame.grid(row=2, column=0, sticky='ew', padx=10, pady=5)
        stages_frame.columnconfigure(1, weight=1)
        
        stages = ["Basecalling", "Alignment", "Variant Calling"]
        for i, stage_name in enumerate(stages):
            # *** STYLE FIX: Use the new 'Panel.TLabel' style ***
            label = ttk.Label(stages_frame, text=f"{stage_name}:", width=15, anchor=tk.E,
                              style='Panel.TLabel')
            label.grid(row=i, column=0, sticky='e', padx=5, pady=8)
            
            pb = ttk.Progressbar(stages_frame, orient="horizontal", length=400,
                                 mode="determinate", style="Horizontal.TProgressbar")
            pb.grid(row=i, column=1, sticky='ew', padx=5, pady=8)
            self.progress_bars[stage_name] = pb
            
            # *** STYLE FIX: Use the new 'Panel.TLabel' style ***
            pl = ttk.Label(stages_frame, text="0%", width=5, anchor=tk.W,
                           style='Panel.TLabel')
            pl.grid(row=i, column=2, sticky='w', padx=5, pady=8)
            self.progress_labels[stage_name] = pl

        # --- 4. Live Data (GENE VISUALIZER) ---
        data_frame = ttk.LabelFrame(self, text="Live Gene Construction (mRNA)",
                                    style='Metallic.TLabelFrame', padding=(5, 5))
        data_frame.grid(row=3, column=0, sticky='nsew', padx=10, pady=5)
        data_frame.rowconfigure(0, weight=1)
        data_frame.columnconfigure(0, weight=1)
        
        # Add a scrollbar
        self.gene_scrollbar = ttk.Scrollbar(data_frame, orient=tk.VERTICAL, style="Vertical.TScrollbar")
        
        # Add the custom GeneVisualizer widget
        self.gene_visualizer = GeneVisualizer(
            data_frame,
            bg=COLOR_TEXT_BOX,
            yscrollcommand=self.gene_scrollbar.set,
            relief=tk.FLAT,
            highlightthickness=0
        )
        
        # Configure scrollbar
        self.gene_scrollbar.config(command=self.gene_visualizer.yview)
        
        # Pack them
        self.gene_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.gene_visualizer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- 5. Status Bar ---
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self, textvariable=self.status_var,
                                    style='Status.Normal.TLabel', anchor=tk.W)
        self.status_bar.grid(row=4, column=0, sticky='ew', ipady=5)
        
        # Initial status updated for mock mode
        if MOCK_MODE:
            self.set_status("Ready. MOCK_MODE is active for visualization.", 'normal')
        else:
            self.set_status("Ready. Select port and press Connect.", 'normal')

    def start_serial_handler(self):
        """Creates and starts the serial handler thread."""
        handler = SerialHandler(self.gui_to_serial_queue, self.serial_to_gui_queue)
        self.serial_handler_thread = threading.Thread(target=handler.run, daemon=True)
        self.serial_handler_thread.start()
        
    def poll_queue(self):
        """Checks the queue for new messages from the serial thread."""
        try:
            # The original implementation is excellent for thread-safe GUI updates
            msg_type, payload = self.serial_to_gui_queue.get_nowait()
            
            if msg_type == 'status':
                status, style = payload
                self.set_status(status, style)
                if 'Connected' in status:
                    self.is_connected = True
                    self.connect_button.config(text="Disconnect", style="Red.TButton")
                    self.port_combo.config(state='disabled')
                    self.baud_entry.config(state='disabled')
                elif 'Disconnected' in status or 'failed' in status:
                    self.is_connected = False
                    self.connect_button.config(text="Connect", style="TButton")
                    # Re-enable port/baud entry
                    self.port_combo.config(state='readonly' if MOCK_MODE else 'normal') 
                    self.baud_entry.config(state='normal')
                    
            elif msg_type == 'data':
                self.process_fpga_message(payload)
                
            elif msg_type == 'log':
                # Can be uncommented for detailed debugging
                # print(f"Serial Log: {payload}")
                pass

        except queue.Empty:
            pass
        finally:
            self.after(100, self.poll_queue) # Poll every 100ms

    def process_fpga_message(self, message):
        """Parses the message from the FPGA and updates the GUI."""
        try:
            parts = message.split(',', 1)
            cmd = parts[0]
            payload = parts[1] if len(parts) > 1 else ""

            if cmd == 'S': # Start: S,Read_ID
                self.set_status(f"Processing new read: {payload}", 'normal')
                self.gene_visualizer.clear() # Clear visualizer for new run
                # Reset all progress bars to 0%
                for stage in ["Basecalling", "Alignment", "Variant Calling"]:
                    self.update_progress(stage, 0)
            
            elif cmd == 'B': # Basecalling: B,Percentage
                self.update_progress("Basecalling", int(payload))
            elif cmd == 'A': # Alignment: A,Percentage
                self.update_progress("Alignment", int(payload))
            elif cmd == 'V': # Variant Calling: V,Percentage
                self.update_progress("Variant Calling", int(payload))
            
            elif cmd == 'D': # Data: D,BaseSequence
                # Feed new bases to the visualizer
                self.gene_visualizer.add_bases(payload)

            elif cmd == 'F': # Finished: F,Message
                self.set_status(f"Job Complete: {payload}. VCF generated.", 'success')
                self.update_progress("Basecalling", 100, 'success') # Ensure all are 100%
                self.update_progress("Alignment", 100, 'success')
                self.update_progress("Variant Calling", 100, 'success')

            elif cmd == 'E': # Error: E,ErrorMessage
                self.set_status(f"FPGA Error: {payload}", 'error')

        except Exception as e:
            # Log parsing errors
            self.set_status(f"GUI Error processing: '{message}'. Details: {e}", 'error')

    def update_progress(self, stage_name, percentage, style_suffix=None):
        """Helper to update a progress bar and its label."""
        if stage_name in self.progress_bars:
            pb = self.progress_bars[stage_name]
            style_name = "Horizontal.TProgressbar"
            
            # Apply success style if explicitly requested or if percentage reaches 100
            if style_suffix == 'success' or percentage >= 100:
                style_name = "Success.Horizontal.TProgressbar"
                percentage = 100 # Clamp at 100
            
            pb.config(value=percentage, style=style_name)
            self.progress_labels[stage_name].config(text=f"{percentage}%")

    def set_status(self, message, style='normal'):
        """Updates the bottom status bar with text and style."""
        self.status_var.set(f" {message}") # Add padding
        if style == 'success':
            self.status_bar.config(style='Status.Success.TLabel')
        elif style == 'error':
            self.status_bar.config(style='Status.Error.TLabel')
        else:
            self.status_bar.config(style='Status.Normal.TLabel')

    def toggle_connection(self):
        """Sends a 'connect' or 'stop' command to the serial thread."""
        if self.is_connected:
            self.gui_to_serial_queue.put(('stop', None))
        else:
            port = self.port_var.get()
            baud_str = self.baud_var.get()
            
            if not port:
                messagebox.showerror("Connection Error", "Please select a serial port.")
                return
                
            if not baud_str.isdigit():
                messagebox.showerror("Connection Error", "Please enter a valid baud rate (e.g., 115200).")
                return
            
            # Send connect command
            self.gui_to_serial_queue.put(('connect', (port, int(baud_str))))

    def on_closing(self):
        """Called when the user closes the window."""
        print("Closing application...")
        if self.serial_handler_thread and self.serial_handler_thread.is_alive():
            # Stop the serial thread gracefully
            self.gui_to_serial_queue.put(('stop', None))
            self.serial_handler_thread.join(timeout=0.5)
        self.destroy()


if __name__ == "__main__":
    app = PipelineMonitor()
    app.mainloop()
