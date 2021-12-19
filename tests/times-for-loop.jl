# @Created by: octave
# @        on: 2021-11-22T11:57:50+00:00
# @Last modified by: octave
# @              on: 2021-11-23T15:19:17+00:00


using LinearAlgebra

t1 = 0.101
t2 = 0.359
t3 = 0.010
t4 = 0.114

p = Diagonal([0.01, 0.4, 0.4, 0.01])

w0 = I - exp(-p*(t1+t2+t3+t4))
w1 = exp(-p*(t4+t3+t2)) * (I-exp(-p*t1)) / w0
w2 = exp(-p*(t4+t3)) * (I-exp(-p*t2)) / w0
w3 = exp(-p*(t4)) * (I-exp(-p*t3)) / w0
w4 = (I-exp(-p*t4)) / w0
