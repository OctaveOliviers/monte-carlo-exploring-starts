# @Created by: octave
# @        on: 2022-03-30T10:11:19+02:00
# @Last modified by: octave
# @              on: 2022-04-06T15:28:15+02:00

using LinearAlgebra


T = [[0 0 1 1];
   [1 1 0 0]]

P1 = [[1 0];
      [0 0];
      [0 1];
      [0 0]]

P2 = [[0 0];
      [1 0];
      [0 1];
      [0 0]]

P3 = [[0 0];
      [1 0];
      [0 0];
      [0 1]]

P4 = [[1 0];
      [0 0];
      [0 0];
      [0 1]]

g = rand()
p = Diagonal(rand(4))
a = normalize(rand(4,1), 1)

(A1, A2, A3, A4) = map(x -> -(I-g*T'*x')*p*inv(I-g*T'*x'), [P1, P2, P3, P4])
(B1, B2, B3, B4) = map(x -> (I-g*T'*x'), [P1, P2, P3, P4])
eigen( a[1]*A1 + a[2]*A2 + a[3]*A3 + a[4]*A4 ).values


# ysstem with positive eigenvalue
begin
    g = 0.95
    p = Diagonal([0.6, 0.2, 0.2, 0.6])
    a = [0.3, 0.1, 0.3, 0.3]

    # g = 0.9250381288350811
    # p = Diagonal([0.608907, 0.187484, 0.182157, 0.520082])
    # a = [0.2707674490127806, 0.061648250647677066, 0.3308323638525128, 0.3367519364870297]
end
