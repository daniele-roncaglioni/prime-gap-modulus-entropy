# Entropy of prime gaps modulo m

## Steps
1) Compute prime numbers up to large number N 
2) Calculate differences between subsequent primes
3) Take those differences modulo M some even number 
4) Compute probability distribution of said differences based on their frequencies 
5) Compute shannon entropy of the distribution 
6) Repeat and plot for various N and M


Note: only even M makes sense as all primes beside 2 are odd and the difference between two odd numbers is always even.


## Result

![entropy](/results/modulo_evens/entropy_curves.png)

Very noticable that as expected for modulo 6 th entropy is lower and seems to decrease the more primes we consider. It is know that prim gaps have the highest probabilities at multipes of 6.


![entropy](/results/modulo_evens/gap_distributions.png)