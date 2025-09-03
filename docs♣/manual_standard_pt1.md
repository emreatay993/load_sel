# **Load SEL Program Documentation**

This document provides a detailed explanation of the Load SEL Program, covering its purpose, capabilities, and user interface.

## **Bölüm 1: Giriş (Chapter 1: Introduction)**

### **1.1 Programın Amacı (Purpose of the Program)**

The **Load SEL Program** (WE MechLoad Viewer) is a specialized engineering software tool designed to streamline the post-processing of mechanical load data. In the product development lifecycle, engineers frequently work with large datasets generated from multi-body dynamic (MBD) simulations or physical prototype testing. This data, representing forces, moments, and accelerations over time or frequency, is often complex and voluminous, making direct interpretation and comparison difficult.

The primary purpose of this program is to serve as an **interactive analysis and data preparation hub**. It provides a graphical user interface (GUI) that empowers engineers to:

* **Rapidly visualize** complex, multi-channel load data without scripting.  
* **Directly compare** different datasets to evaluate design changes or validate simulations against physical test results.  
* **Process and condition** signals using integrated filtering and windowing tools.  
* **Automate the export** of this data into a format directly usable by Finite Element Analysis (FEA) software, specifically Ansys Mechanical.

**Analogy:** Think of the program as a **digital workbench for your simulation data**. Instead of manually sorting through text files and writing scripts, you have a set of specialized tools (the tabs) that allow you to probe, measure, compare, and prepare your data for the next stage of analysis, all within one integrated environment.

### **1.2 Programın Kabiliyetleri (Capabilities of the Program)**

The program is equipped with a focused set of capabilities tailored to the engineering workflow:

* **Dual-Domain Data Handling:** The software automatically detects whether the input data is in the **time-domain** (e.g., Force vs. Time) or **frequency-domain** (e.g., Amplitude vs. Frequency) and adjusts the available tools and plots accordingly.  
* **Multi-Dataset Comparison:** A core feature is the ability to load and overlay multiple datasets seamlessly. This is facilitated by a directory tree view, allowing for quick comparisons between different simulation runs or data folders.  
* **Specialized Visualization Tabs:** The interface is logically divided into tabs, each offering a unique perspective on the data:  
  * **Single Data:** For in-depth analysis of one data channel at a time.  
  * **Interface Data:** For viewing all six degrees of freedom (Fx, Fy, Fz, Mx, My, Mz) for a single connection point or interface.  
  * **Part Loads:** For grouping and analyzing loads based on component names or locations (e.g., all vertical forces on a "Left-Hand-Side" component).  
* **Integrated Signal Processing:** For time-domain data, the program provides essential on-the-fly signal conditioning tools:  
  * **Low-Pass Filtering:** To remove unwanted high-frequency noise.  
  * **Data Sectioning:** To isolate and focus on a specific time interval of interest.  
  * **Tukey Windowing:** To prepare signals for Fourier analysis by minimizing spectral leakage.  
* **Automated Ansys Export:** The program can generate all necessary files for applying complex load histories in Ansys Mechanical. It creates both the formatted data table (.dat) and the corresponding APDL command script (.inp) for Transient Structural or Harmonic Response analyses.

## **Bölüm 2: Veri Okuma (Chapter 2: Data Reading)**

### **2.1 Veri Formatı ve Klasör Yapısı (Data Format and Folder Structure)**

The program requires a specific and consistent data structure to correctly parse and interpret the load files. This standardization is crucial for the automated features of the application to function reliably.

Each dataset must be contained within its own folder. Inside this folder, the program expects to find two specific types of files:

1. **full.pld:** This is the **raw numerical data file**. The program expects this to be a text file where data columns are separated by the pipe (|) delimiter. This file should contain only the numerical values of the load channels.  
2. **max.pld:** This is the **header file**. It is also a pipe-delimited text file and contains the corresponding names (labels) for each data column present in the full.pld file. The order of the headers in this file must match the order of the data columns.

**Analogy:** The full.pld file is like a large spreadsheet table filled with numbers. The max.pld file acts as the **frozen header row** at the top of that spreadsheet. Without this header row, the program would see columns of numbers but wouldn't know if a column represents "Force in X-Direction" or "Moment about Z-Axis". The folder structure ensures that the data and its description are always kept together.

### **2.2 Veri Okutma (Reading Data)**

There are two primary methods for loading data into the application:

1\. Initial Data Loading:  
This method is used to load the first, or primary, dataset when you launch the program.

1. From the main menu bar at the top of the window, select **File \-\> Open New Data**.  
2. A system file dialog will appear. Navigate to the directory that contains your full.pld and max.pld files.  
3. Select the **folder itself** (not the files inside) and click **"Select Folder"**.

The program will then load the data, populate the channel selectors in all tabs, and display a default plot in the "Single Data" tab.

2\. Loading Data for Comparison:  
This method is used to load one or more additional datasets to overlay and compare against the initial one.

1. After loading your initial dataset, direct your attention to the **Directory Tree** panel on the left side of the window.  
2. This tree will display the parent directory and all sibling folders of the dataset you initially loaded.  
3. To load an additional dataset, simply **click the checkbox** next to the desired folder name in the tree.

The program will instantly load the data from the checked folder and overlay its plots on top of the existing ones in all relevant tabs, color-coding them for easy differentiation. You can check and uncheck multiple folders to dynamically add and remove datasets from the comparison.

## **Bölüm 3: Arayüz Tanıtımı (Chapter 3: Interface Introduction)**

### **3.1 "Single Data" Sekmesi (Single Data Tab)**

This is the most fundamental analysis tab, designed for the detailed inspection of a **single data channel**. It provides tools for both time and frequency domain analysis.

* **Column Selector:** A dropdown menu at the top-left that lists all available data channels from the header file. This is the primary control for this tab; selecting a channel immediately updates all plots.  
* **Main Plot:** A large plotting area that displays the magnitude of the selected channel. For time-domain data, it shows Amplitude vs. Time. For frequency-domain data, it shows Amplitude vs. Frequency.  
* **Phase Plot:** This plot appears directly below the main plot **only for frequency-domain data**. It displays the phase angle of the selected channel versus frequency.  
* **Time-Domain Specific Tools:** A set of checkboxes and inputs appear on the right side of the selector bar only for time data:  
  * **Show Spectrum Plot:** When checked, this displays a third plot at the bottom showing a time-frequency spectrogram, which is useful for identifying how the frequency content of a signal changes over time.  
  * **Apply Low-Pass Filter:** Enables a Butterworth filter to smooth the signal. When checked, input fields for **Cutoff Frequency** and **Filter Order** appear.

### **3.2 "Interface Data" Sekmesi (Interface Data Tab)**

This tab is designed to provide a holistic view of all loads acting at a **single physical connection point**, which is referred to as an "interface." The program automatically groups channels together based on a common prefix in their names (e.g., I1\_Fx, I1\_Fy are grouped under the I1 interface).

* **Interface Selector:** A dropdown menu that lists all the unique interfaces identified from the data channel headers.  
* **Plot:** A single plot that displays all channels belonging to the selected interface simultaneously. For example, selecting interface I12 might show six distinct lines on the plot: I12 Fx, I12 Fy, I12 Fz, I12 Mx, I12 My, and I12 Mz. This is extremely useful for understanding the complete loading condition at a specific bolt, weld, or connection.

### **3.3 "Part Loads" Sekmesi (Part Loads Tab)**

This tab offers a more flexible way to group and view data based on common keywords within the channel names. It is also the primary control center for **data processing and exporting to Ansys**.

* **Side Filter Selector:** A dropdown menu that is populated with common identifiers found in the channel names, which typically represent a component, side, or location (e.g., "LHS", "Engine Mount", "Front").  
* **Component Filter Selector:** A second dropdown menu that allows you to filter for a specific load component (e.g., "Fx", "My", "Accel").  
* **Plot:** The plot displays all data channels that match *both* the Side Filter and the Component Filter criteria. For example, you could view all "Fx" forces on the "LHS".  
* **Data Processing Controls (Time Domain Only):**  
  * **Apply Data Section:** Allows you to crop the data to a specific time interval by setting a Min and Max time.  
  * **Apply Tukey Window:** Applies a tapered cosine window to the signal.  
* **Export to Ansys Button:** This is the key function of this tab. After you have filtered and processed your data as desired, clicking this button initiates the process of creating the Ansys input files.