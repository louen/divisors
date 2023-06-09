import numpy as np
import itertools

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



def divisors(n):
    if n == 0:
        return []
    if n == 1:
        return [1]
    """ Compute all divisors of n, including 1 and itself"""
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
    if n == 1:
        return [1]
    """Return all numbers less than n that are coprimes with n"""
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
    d = divisors (N)
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

    assert divisors(1) == [1]
    assert divisors(25) == [1,5,25]




def pairs_number_divisors(N,k):
    # Naive count of pairs
    return [ (x,y) for  (x,y) in itertools.product(range(1,N+1), repeat=2) if len(divisors(np.gcd(x,y))) == k]

tests()


for N in [1,2,3,4]:
    k = 1
    p = pairs_number_divisors(N,k)
    n = len(p)
    print(f'There are {n} pairs of numbers ({n/N**2:.2%}) between 1 and {N} with {k} common divisors')
    print(p)

    if k == 1:
        check = np.sum([len(coprimes(i)) for i in range(1,N+1)])
        print(f'{N} : {n} | {2*check-1}')
        assert n == 2 * check - 1
