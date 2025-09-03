# **Load SEL Program Documentation (Part 2\)**

## **Bölüm 4: Veri Karşılaştırma (Chapter 4: Data Comparison)**

The ability to compare datasets is a central feature of the Load SEL Program. This chapter details the two specialized tabs designed for this purpose. Comparison can be done either by loading multiple folders via the Directory Tree (as explained in Chapter 2.2) or by loading a separate, distinct comparison file.

### **4.1 Karşılaştırma Verisi Okuma (Reading Comparison Data)**

This method is intended for when you want to compare your primary dataset against another one that is located in a different, non-sibling directory. It loads a secondary dataset into memory specifically for comparison purposes.

1. First, ensure you have already loaded a primary dataset using **File \-\> Open New Data**.  
2. Navigate to the **"Compare Data"** tab.  
3. Click the **"Select Data to Compare"** button located in the top control bar.  
4. A system file dialog will open. Navigate to the folder of the secondary dataset you wish to analyze.  
5. Select the folder and confirm. A message box will appear confirming that the comparison data has been loaded successfully.

Once loaded, the channel selectors in the "Compare Data" and "Compare Part Loads" tabs will be populated, and the comparison plots will become active.

### **4.2 "Compare Data" Sekmesi (Compare Data Tab)**

This tab is designed for a direct, **one-to-one comparison** of a single data channel between your primary and secondary datasets. It is the ideal tool for detailed checks where you need to see how a specific channel differs between two cases.

* **Column Selector:** This dropdown lists all the channels from your **primary** dataset (the one you loaded first).  
* **Column to Compare:** This dropdown lists all the channels from your **secondary** (comparison) dataset.  
* **Plot:** The plot area displays both selected signals overlaid on the same axes.  
  * The primary signal is typically plotted as a solid line.  
  * The comparison signal is plotted as a dashed line for clear visual distinction.

This view makes it incredibly easy to spot differences in magnitude, phase shifts (for frequency data), or timing of peaks (for time data).

### **4.3 "Compare Part Loads" Sekmesi (Compare Part Loads Tab)**

This tab extends the comparison functionality to **groups of channels**, combining the filtering logic of the "Part Loads" tab with the dual-dataset view of the "Compare Data" tab. It allows you to compare the behavior of entire components or sub-assemblies between two different datasets.

* **Side Filter Selector:** Filters for a component or location identifier (e.g., "LHS") across **both** datasets.  
* **Component Filter Selector:** Filters for a specific load component (e.g., "Fx") across **both** datasets.  
* **Plot:** The plot displays all channels that match the filter criteria from both the primary and comparison datasets. All signals from the primary dataset are shown as solid lines, and all corresponding signals from the comparison dataset are shown as dashed lines. This provides a powerful, high-level view of how a design change or different load case affects a whole group of related loads.

## **Bölüm 5: Zaman Domeni Sinyali Oluşturma (Chapter 5: Creating Time Domain Signal)**

### **5.1 "Time Domain Rep." Sekmesi ("Time Domain Rep." Tab)**

This is a specialized analysis tab that is **only visible when you have loaded frequency-domain data**. Its purpose is to help engineers visualize what a steady-state harmonic signal would look like in the time domain, based on the information from a single frequency point.

It effectively performs a reverse calculation, reconstructing a time-domain sine wave from the magnitude and phase data available in the frequency domain.

* **Data Point Selector:** A dropdown menu that is populated with all the unique frequency points from your dataset (e.g., 5 Hz, 10 Hz, 15 Hz...). You must select a single frequency point to activate the plot.  
* **Plot:** The plot shows a reconstructed time-domain signal for all data channels, calculated over a full 360-degree cycle. Each signal is a perfect sine wave whose **amplitude** and **phase shift** are determined by the magnitude and phase values from your data at the selected frequency. This is useful for understanding the relative motion and timing between different load channels at a specific harmonic frequency.  
* **Interval Selector:** Allows you to choose the angular interval (in degrees) for which you want to see data points on the plot.  
* **Extract Data Button:** This function allows you to export the reconstructed time-domain data to a .csv file. The exported file will contain a table of the load values at each angular interval, which can be useful for further custom analysis or reporting.