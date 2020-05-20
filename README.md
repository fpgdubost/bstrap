Bootstrapping for hypothesis testing and computation of confidence interval.

You want to compare two methods and claim that one is better than the other?
Bootstrapping is an easy method to compute statistics over your metrics. It has the advantage of being very versatile. 
I have implemented in this repository:
-bootstrapping for computation of confidence interval
-bootstrapping for hypothesis testing (claim that one method is better than another for a given metric)

Keep in mind: non-overlapping confidence intervals means that there is a significant statistical difference. Overlapping CI does not mean that there is no significant statistical difference. To verify this further, you will need to compute the bootstrap hypothesis testing and check the p-value.

To use this code, you need to:

1.implement your own custom_metric function
2.change to code to load your data
3.check that your estimates (CI bounds and p-value) are stable over several runs of the bootstrapping method. If it is not, increase nbr_runs. 

Enjoy!

I am a researcher in deep learning, medical image analysis and neurology. Check out my work!
https://scholar.google.com/citations?user=_yNBmx8AAAAJ&hl=fr

Wanna watch some cool videos?
https://www.zmediacorp.org/

Reference:
Efron, B. and Tibshirani, R.J., 1994. An introduction to the bootstrap. CRC press.

PS: I am not responsible for the use you make of this code. Always double check your results.
