# @Created by: OctaveOliviers
# @        on: 2021-04-01T15:07:04+02:00
# @Last modified by: OctaveOliviers
# @              on: 2021-04-06T13:08:42+02:00


using ProgressBars

include("mces.jl")
include("mdp.jl")
include("utils.jl")

@debug "Set seed" Random.seed!(Dates.day(today()))

# # structure of markov tree
# S = [[1 0 0 0 0];
#      [1 0 0 0 0];
#      [0 1 0 0 0];
#      [0 1 0 0 0];
#      [0 0 1 0 0];
#      [0 0 1 0 0];
#      [0 0 0 1 0];
#      [0 0 0 0 1]]
# # transition probabilities
# T = [[0 0 0 0 1 0 0 0];
#      [1 0 0 0 0 0 0 0];
#      [0 0 1 0 0 0 0 0];
#      [0 1 0 1 0 0 1 0];
#      [0 0 0 0 0 1 0 1]]
# # rewards of each state-action
# r = [0, -10, 0, -10, 0, 10, 0, 0]
# # optimal q values and policy
# q_opt = [8.1, -10, 9, -10, 7.29, 10, 0, 0]
# P_opt = compute_policy(S, q_opt)


# test is obtain limit cycle when alpha_k remains constant
if true

    num_s = 3+convert(Int64, ceil(5*rand()))
    num_sa = convert(Int64, 3*num_s + ceil(10*rand()))

    # discount factor
    gam = 0.9*rand()
    # prior probability of starting episode in each state-action
    p = rand(Float64, num_sa)
    p = p/sum(p)

    S, T, P_opt, r, q_opt, gam = generate_mdp(num_sa, num_s, discount=gam)

    q = monte_carlo_exploring_start(S, T, r, p, gam, q_opt, num_epi = 1e4, max_len_epi = 1e2)



end


# test contraction values
if false

    num_s = 3+convert(Int64, ceil(5*rand()))
    num_sa = convert(Int64, 3*num_s + ceil(10*rand()))

    # discount factor
    gam = 0.9*rand()
    # prior probability of starting episode in each state-action
    p = rand(Float64, num_sa)
    p = p/sum(p)

    S, T, P_opt, r, q_opt, gam = generate_mdp(num_sa, num_s, discount=gam)

    q, c = monte_carlo_exploring_start(S, T, r, p, gam, q_opt, num_epi = 1e4, max_len_epi = 1e2)

    @info q q_opt

    # store contractions
    open("contractions.txt", "a") do io
        println(io, c)
    end


end


# test on weights
if false

    num_s = 10+convert(Int64, ceil(5*rand()))
    num_sa = convert(Int64, 3*num_s + ceil(20*rand()))

    # discount factor
    gam = 0.9*rand()
    # prior probability of starting episode in each state-action
    p = rand(Float64, num_sa)
    p = p/sum(p)

    S, T, P_opt, r, q_opt, gam = generate_mdp(num_sa, num_s, discount=gam)

    @info "Parameters are" num_s num_sa P_opt'*q_opt gam

    q1 =100*rand(Float64, num_sa)
    P1 = compute_policy(S, q1)
    stp, D1 = step(P1, T, r, p, q1, gam, 500, return_diag=true)

    q2 = policy_q(P1, T, r, gam)
    P2 = compute_policy(S, q2)
    stp, D2 =  step(P2, T, r, p, q2, gam, 500, return_diag=true)

    q3 = q2 + stp./(D1+D2)
    P3 = compute_policy(S, q3)
    stp, D3 =  step(P3, T, r, p, q3, gam, 500, return_diag=true)

    q4 = q3 + stp./(D1+D2+D3)
    P4 = compute_policy(S, q4)
    stp, D4 =  step(P4, T, r, p, q4, gam, 500, return_diag=true)

    q5 = q4 + stp./(D1+D2+D3+D4)
    P5 = compute_policy(S, q5)
    stp, D5 =  step(P5, T, r, p, q5, gam, 500, return_diag=true)

    q6 = q5 + stp./(D1+D2+D3+D4+D5)
    P6 = compute_policy(S, q6)
    stp, D6 =  step(P6, T, r, p, q6, gam, 500, return_diag=true)

    q7 = q6 + stp./(D1+D2+D3+D4+D5+D6)
    P7 = compute_policy(S, q7)
    stp, D7 =  step(P7, T, r, p, q7, gam, 500, return_diag=true)

    q_p1 = q2
    q_p2 = policy_q(P2, T, r, gam)
    q_p3 = policy_q(P3, T, r, gam)
    q_p4 = policy_q(P4, T, r, gam)
    q_p5 = policy_q(P5, T, r, gam)
    q_p6 = policy_q(P6, T, r, gam)
    q_p7 = policy_q(P7, T, r, gam)

    # check inequalities
    # # k = 2
    # W = D2./(D1+D2)
    # P3'*W.*(P2'*q_p2 - P3'*q2) .<= P2'*W.*(P2'*q_p2-P2'*q2) + (P2'*q2-P3'*q2)
    # # k = 3
    # W = D3./(D1+D2+D3)
    # P4'*W.*(P3'*q_p3 - P4'*q3) .<= P3'*W.*(P3'*q_p3-P3'*q3) + (P3'*q3-P4'*q3)
    # # k = 4
    # W = D4./(D1+D2+D3+D4)
    # P5'*W.*(P4'*q_p4 - P5'*q4) .<= P4'*W.*(P4'*q_p4-P4'*q4) + (P4'*q4-P5'*q4)
    # # k = 5
    # W = D5./(D1+D2+D3+D4+D5)
    # P6'*W.*(P5'*q_p5 - P6'*q5) .<= P5'*W.*(P5'*q_p5-P5'*q5) + (P5'*q5-P6'*q5)

    # k=2
    W = D2./(D1+D2)
    W_ = sum((S' .* (P2'*W))', dims=2 )
    q_ = sum((S' .* (P2'*q2))', dims=2)
    W .*(q_p2-q2) .<= W_.*(q_p2-q_) + (q_-q2)

    # k=4
    W = D5./(D1+D2+D3+D4+D5)
    W_ = sum((S' .* (P5'*W))', dims=2 )
    q_ = sum((S' .* (P5'*q5))', dims=2)
    W .*(q_p5-q5) .<= W_.*(q_p5-q_) + (q_-q5)
    # k=5
    W = D5./(D1+D2+D3+D4+D5)
    W_ = sum((S' .* (P5'*W))', dims=2 )
    q_ = sum((S' .* (P5'*q5))', dims=2)
    W .*(q_p5-q5) .<= W_.*(q_p5-q_) + (q_-q5)
    # k=6
    W = D6./(D1+D2+D3+D4+D5+D6)
    W_ = sum((S' .* (P6'*W))', dims=2 )
    q_ = sum((S' .* (P6'*q6))', dims=2)
    W .*(q_p6-q6) .<= W_.*(q_p6-q_) + (q_-q6)


    # set up parameters
    A1, A2, A3,A4 = P1*T, P2*T, P3*T, P4*T
    W21, W22 = D1./(D1+D2), D2./(D1+D2)
    W31, W32, W33 = D1./(D1+D2+D3), D2./(D1+D2+D3), D3./(D1+D2+D3)
    # tests
    println("\nAt k=2")
    println("Compare with A3 ", W21 .* (A1'*q_p1) + W22 .* (A2'*q_p2) .<= A3' * (W21.*q_p1 + W22.*q_p2) )
    println("Compare with A2 ", W21 .* (A1'*q_p1) + W22 .* (A2'*q_p2) .<= A2' * (W21.*q_p1 + W22.*q_p2) )

    println("\nAt k=3")
    println("Compare with A4 ", W31 .* (A1'*q_p1) + W32 .* (A2'*q_p2) + W33 .* (A3'*q_p3) .<= A4' * (W31.*q_p1 + W32.*q_p2 + W33.*q_p3) )
    println("Compare with A3 ", W31 .* (A1'*q_p1) + W32 .* (A2'*q_p2) + W33 .* (A3'*q_p3) .<= A3' * (W31.*q_p1 + W32.*q_p2 + W33.*q_p3) )

end

# test size of weights
if false
    # # create MDP
    # num_s = 3+convert(Int64, ceil(5*rand()))
 #    num_sa = convert(Int64, 2*num_s + ceil(20*rand()))

 #    # discount factor
 #    gam = 0.9*rand()
 #    # prior probability of starting episode in each state-action
 #    p = rand(Float64, num_sa)
 #    p = p/sum(p)

 #    S, T, P_opt, r, q_opt, gam = generate_mdp(num_sa, num_s, discount=gam)
    # @info "Parameters are" num_s num_sa P_opt'*q_opt gam

    gam = 0.9 ;
    p = [0.1, 0.2, 0.1, 0.4, 0.2, 0, 0, 0]
    S = [[1 0 0 0 0 0];
         [0 1 0 0 0 0];
         [0 1 0 0 0 0];
         [0 0 1 0 0 0];
         [0 0 1 0 0 0];
         [0 0 0 1 0 0];
         [0 0 0 0 1 0];
         [0 0 0 0 0 1]]
    T = [[0 0 0 0 0 0 0 0];
         [1 0 0 .8 0 0 0 0];
         [0 0 1 0 0 0 0 0];
         [0 1 0 0 0 1 0 0];
         [0 0 0 0 1 0 1 0];
         [0 0 0 .2 0 0 0 1]]
    r = [0., 10., 0., -10., -20., 0., 0., 0.] ;
    P = [[1 0 0 0 0 0];
         [0 0 0 0 0 0];
         [0 1 0 0 0 0];
         [0 0 1 0 0 0];
         [0 0 0 0 0 0];
         [0 0 0 1 0 0];
         [0 0 0 0 1 0];
         [0 0 0 0 0 1]]

    q = monte_carlo_exploring_start(S, T, r, p, gam, num_epi = 1e6, max_len_epi = 2*1e1)
    # q = expected_mces(S, T, r, p, gam, num_epi = 1e3, max_len_epi = 4*1e2)

    @info q
end

# test mote carlo algorithm
if false

    num_test = 1

    for n = 1:num_test
        println("\nTest ", n, " of ", num_test)

        num_s = 5  #3+convert(Int64, ceil(5*rand()))
        num_sa = 15 #convert(Int64, 2*num_s + ceil(20*rand()))

        # discount factor
        gam = 0.9*rand()
        # prior probability of starting episode in each state-action
        p = rand(Float64, num_sa)
        p = p/sum(p)

        S, T, P_opt, r, q_opt, gam = generate_mdp(num_sa, num_s, gam=gam)

        @info "Parameters are" num_s num_sa P_opt'*q_opt gam

        # q = monte_carlo_exploring_start(S, T, r, p, gam)
        q = expected_mces(S, T, r, p, gam, num_epi = 32000, max_len_epi = 2*1e2)
        P = compute_policy(S,q)

        println("Found correct policy: ", P == P_opt)
    end

    # println("\nvalues are       $(round.(P'*q, digits=3))")
    # println("\nvalues should be $(round.(P_opt'*q_opt, digits=3))")
end

# test inequalities on weights
if false

    num_sa = 30
    num_s = 10

    # discount factor
    gam = 0.9
    # prior probability of starting episode in each state-action
    p = rand(Float64, num_sa)
    p = p/sum(p)

    S, T, P_opt, r, q_opt, gam = generate_mdp(num_sa, num_s)

    q0 = 100*rand(num_sa)
    P1 = compute_policy(S, q0)
    pot, val, q1 = compute_potential(P1, T, r, p, q0, gam, 400, return_q_opt=true) ;

    P2 = compute_policy(S, q1)
    pot, val, q2 = compute_potential(P2, T, r, p, q1, gam, 400, return_q_opt=true) ;

    W = Diagonal(rand(num_sa))
    q3 = W*q1 + (I-W)*q2

    P3 = compute_policy(S, q3)
    # pot, val, q2 = compute_potential(P3, T, r, p, q1, gam, 400, return_q_opt=true) ;

    if P3 != P2
        @info "new policy"
    end

    println( W*T'*P2'*q1 + (I-W)*T'*P2'*q2 .<= T'*P3'*(W*q1 + (I-W)*q2) )
end

# println(q)
# println("Computed q-values are $q")
# println("Should be close to $q_opt")
# println("computed vs actual q-values")
# println("q values are $(round.([q,q_opt], digits=3))")
