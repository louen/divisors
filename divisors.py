import itertools
import numpy as np
import matplotlib.pyplot as plt

# Integer value of the square root
intsqrt = lambda x : int(np.floor(np.sqrt(x)))
def primes(N):
    """Compute primes numbers from 1 to N (included) with the sieve of erathostenes"""
    isPrime = [True] * N
    isPrime[0] = False # 1 is not prime
    for p in range(2, intsqrt(N)+1):
        if isPrime[p-1]:
            for q in range(N // p - p + 1):
                isPrime[p*(p+q) - 1] = False

    return [ n+1 for n,v in enumerate(isPrime) if v]

def primefactors(n, prime_cache=None):
    """Prime factors of n, without exponents"""
    if not prime_cache:
        prime_cache = primes(n)
    return [p for p in prime_cache if n % p == 0]

def sqprimefactors(n, prime_cache = None):
    """Prime numbers whose square divides n"""
    if not prime_cache:
        prime_cache = primes(intsqrt(n))
    return [ p for p in prime_cache if n % (p*p) == 0]

def divisors(n):
    """Returns all divisors of n, (including 1 and n)"""
    if n == 0:
        return []
    if n == 1:
        return [1]
    start = []
    end = []
    max = intsqrt(n)
    for i in range (1, max):
        if n % i == 0:
            start.append(i)
            end.append(n // i)
    if max * max == n: # Handle perfect squares
        start.append(max)
    elif n % max == 0:
        start.append(max)
        end.append(n//max)
    return start + list(reversed(end))


def coprimes(n):
    """Return all numbers less than n that are coprimes with n"""
    if n == 1:
        return [1]
    return [i for i in range(1,n) if np.gcd(i,n) == 1]

### tests
def tests():
    ### primes
    N = 5432
    p = primes(N)
    assert len(p) == 717
    for x in p:
        for d in range(2,intsqrt(x)):
            assert x % d != 0
        assert len(divisors(x)) == 2

    ### divisors

    N = 8128 # perfect number
    d = divisors(N)
    assert d[0] == 1 and d[-1] == N
    assert np.sum(d) == 2 * N
    # Amicable numbers
    N1 = 17296
    N2 = 18416
    assert np.sum(divisors(N1)) == N1 + N2
    assert np.sum(divisors(N2)) == N1 + N2

    assert len(coprimes(1)) == 1
    assert len(coprimes(97)) == 96
    assert len(coprimes(3456)) == 1152

    assert primefactors(1) == []
    assert primefactors(5) == [5]
    assert primefactors(27) == [3]
    assert primefactors(1024) == [2]
    assert primefactors(3*5*7*7*2*2*2*97) == [2,3,5,7,97]

    assert divisors(1) == [1]
    assert divisors(25) == [1,5,25]


### Count pairs with  k common divisors

def pairs_number_divisors(N,k):
    # Naive count of pairs
    return [ (x,y) for  (x,y) in itertools.product(range(1,N+1), repeat=2) if len(divisors(np.gcd(x,y))) == k]


def P1(N):
    # formula for k=1
    return 2*np.sum([len(coprimes(n)) for n in range(1,N+1)]) - 1

def P2(N):
    # formula for k =2
    pr = primes(N)
    s = 0
    for n in range(1,N+1):
        for p in primefactors(n,pr):
            s += len(coprimes(n//p))
    return 2*s - len(pr)

def P3(N):
    # Formula for k = 3
    pr = primes(intsqrt(N))
    s = 0
    for n in range(1,N+1):
        for p in sqprimefactors(n,pr):
            s+= len(coprimes(n//(p**2)))
    return 2*s - len(pr)

formulas = [P1,P2,P3]

# Print
for k in [1,2,3]:
    for N in [10, 29, 100, 200,1000]:
        s = formulas[k-1](N)
        print(f'There are {s} pairs of numbers ({s/N**2:.2%}) between 1 and {N} with {k} common divisors')


# Plot
for k in [1,2,3]:
    x = np.logspace(1,4,num=40)
    x = np.floor(x).astype(int)
    y = np.vectorize(formulas[k-1])(x) / (x**2)

    plt.plot(x,y)
    #plt.ylim((0,1))
    plt.show()
