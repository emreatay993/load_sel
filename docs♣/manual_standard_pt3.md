# **Load SEL Program Documentation (Part 3\)**

## **Bölüm 6: Ansys'e Veri Aktarımı (Chapter 6: Data Transfer to Ansys)**

### **6.1 Ansys'e Veri Aktarma (Transferring Data to Ansys)**

This is one of the program's most powerful features, designed to bridge the gap between data analysis and Finite Element Analysis (FEA). It automates the creation of correctly formatted input files for applying complex load histories in Ansys Mechanical. This functionality is exclusively managed from the **"Part Loads"** tab.

**Workflow:**

1. **Navigate to the "Part Loads" Tab:** This is the control center for the export process.  
2. **Filter Your Data:** Use the **Side Filter** and **Component Filter** selectors to isolate the specific set of loads you want to apply in your FEA model. For example, you might select all forces and moments (Fx, Fy, Fz, Mx, My, Mz) for the "LHS\_Engine\_Mount".  
3. **(Optional) Process Your Data:** If you are working with time-domain data, you can apply processing before export:  
   * Use the **Apply Data Section** feature to crop the time signal to a relevant event.  
   * Use the **Apply Tukey Window** feature to taper the signal, which is good practice for subsequent frequency-domain analysis in Ansys.  
4. **Initiate the Export:** Click the **"Export to Ansys"** button.  
5. **Confirm Selection:** A dialog box will appear, asking you to confirm the parts (sides) you wish to include in the export. This allows for a final check before file creation.

**Generated Files:**

Upon confirmation, the program will generate and save two critical files in its root directory:

1. **A Tabular Data File (.dat):** This is a text file containing a precisely formatted table of your load data. The first column is either Time or Frequency, and subsequent columns are the load values for the channels you selected. The values are automatically converted to a consistent unit system expected by Ansys (e.g., forces are multiplied by 1000).  
2. **An APDL Script File (.inp):** This is an Ansys Parametric Design Language script. This script contains all the commands Ansys needs to:  
   * Define tables for each load channel.  
   * Read the data from the .dat file into these tables.  
   * Apply these tables as loads (e.g., as forces or moments) to the appropriate nodes or geometric entities in your model.

This automated process eliminates the tedious and error-prone task of manually creating load tables and scripts, significantly accelerating the simulation workflow.

## **Bölüm 7: Ayarlar (Chapter 7: Settings)**

### **7.1 "Settings" Sekmesi ("Settings" Tab)**

This tab provides global controls to customize the visual appearance of all plots throughout the application. Changes made here will instantly apply to every plot in every tab, allowing you to tailor the viewing experience to your preference or for reporting purposes.

**Available Customization Options:**

* **Show Legend:** A checkbox to toggle the visibility of the plot legend. Disabling it can help de-clutter the view when many channels are plotted.  
* **Show Grid:** A checkbox to toggle the visibility of the background grid on all plots.  
* **Plot Theme Selector:** A dropdown menu that allows you to change the entire color scheme of the plots. This includes the background color, grid lines, and the sequence of colors used for the data lines.  
* **Legend Position:** A dropdown menu to move the legend to different corners of the plot area (e.g., Top Right, Bottom Left), which is useful if the legend is obscuring an important part of the data.  
* **Rolling Min/Max (Time Domain Only):** When checked, this feature calculates and displays the rolling minimum and maximum values of the signal over time, providing a clear envelope of the signal's peaks.

**Keyboard Shortcuts:**

For quick access to the most common settings, two keyboard shortcuts are available from any tab:

* **L Key:** Toggles the legend visibility on and off.  
* **K Key:** Cycles the legend's position through the available corners.

These settings provide a simple yet effective way to manage the visual presentation of your data, ensuring clarity during analysis and generating professional-looking plots for reports and presentations.