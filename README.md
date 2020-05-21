<h1>Bootstrapping for hypothesis testing and computation of confidence interval.</h1>

You want to compare two methods and claim that one is better than the other?
Your metric is more complex than simply averaging results for each samples?
Bootstrapping is an easy method to compute statistics over your custom metrics. It has the advantage of being very versatile. 
I have implemented in this repository:
<ul>
  <li>Bootstrapping for computation of confidence interval</li>
  <li>bootstrapping for hypothesis testing (claim that one method is better than another for a given metric)</li>
</ul>

Keep in mind: non-overlapping confidence intervals means that there is a significant statistical difference. Overlapping CI does not mean that there is no significant statistical difference. To verify this further, you will need to compute the bootstrap hypothesis testing and check the p-value.

To use this code, you need to:

<ol>
  <li>Implement your own custom_metric function.</li>
  <li>Change to code to load your data.</li>
  <li>Check that your estimates (CI bounds and p-value) are stable over several runs of the bootstrapping method. If it is not, increase nbr_runs.</li>
</ol>

Enjoy!

<b>Reference:</b><br/>
Efron, B. and Tibshirani, R.J., 1994. An introduction to the bootstrap. CRC press.

----------------------------------------------------------------------------------------------------------------------------

I am a researcher in deep learning, medical image analysis and neurology. Check out my work!<br/>
https://scholar.google.com/citations?user=_yNBmx8AAAAJ&hl=fr

Wanna watch some cool videos?<br/>
https://www.zmediacorp.org/



<i>PS: I am not responsible for the use you make of this code. Always double check your results.</a>
