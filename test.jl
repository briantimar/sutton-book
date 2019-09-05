include("dp.jl")

# one-dimensional gridworld
states = collect(1:5)
N = length(states)
#two actions, left and right
rewards = -1.0 * ones(N, 2)
rewards[end, :] .= 0

#defines the environmental dynamics
function getsuccessors(state::Int, action::Int, N::Int)
    if state ==1 && action == 1 
        successors = 1:1
    elseif state==N 
        successors = N:N
    else
        inext = 2 * action - 3 + state
        successors = inext:inext
    end
    probs = [1.0]
    return successors, probs
end

getsuccessors(state::Int, action::Int) = getsuccessors(state, action, N)

# set up a deterministic policy
# pol = Policy(rand((1, 2), N))

getactions(state) = 1:2
getreward(state, action) = rewards[state, action]

gamma = 1.0
maxiter=100

# values, pol = policyiteration(pol, getsuccessors, rewards,gamma=gamma,niter=niter,neval=neval)
values = valueiteration(states, getsuccessors, getactions, getreward, gamma=gamma,maxiter=maxiter, tol=1e-3)