# File: app/tooltips.py

"""A central repository for all UI tooltip strings."""

SPECTRUM_SLICES = """
<b>Controls the Time vs. Frequency Resolution Trade-off.</b><br><br>
The total time signal is divided into this many segments (slices)
to see how the frequency content changes over time.<br><br>
&#8226; <b>More Slices:</b> Better <i>time resolution</i> (pinpoint <b>when</b> an
  event occurs), but lower frequency precision.<br>
&#8226; <b>Fewer Slices:</b> Better <i>frequency precision</i> (distinguish
  between close frequencies), but vaguer timing.<br><br>
A good starting point is 400.
"""