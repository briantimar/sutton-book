include("dp.jl")

# define environment succession and reward functions
maxstate=100
pheads=.4
states =collect(0:maxstate)
index(state) = state + 1
stateindices = index.(states)

getactions(state) = 0:min(state-1, maxstate-(state-1))
betev(action, pheads) = pheads * action - (1-pheads) * action

function getsuccessors(state, action)
    #largest /smallest possible dollar amounts
    state == index(maxstate) && return [state], [1.0]
    statecapital = state-1
    low = max(0, statecapital-action)
    high = min(maxstate, statecapital+action)
    #corresponding state indices
    return [index(low), index(high)], [1-pheads, pheads]
end

function getreward(state, action)
    state == index(maxstate) && return 0
    state + action < index(maxstate) && return 0
    return pheads * 1.0
end

gamma=1.0
maxiter=100
values = valueiteration(stateindices, getsuccessors, getactions, getreward, gamma=gamma, maxiter=maxiter)
