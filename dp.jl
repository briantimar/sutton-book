using Distributions: Categorical

" Defines a deterministic policy"
struct Policy
    actions::Vector{Int}
end

(pol::Policy)(state::Int) = pol.actions[state]

"Expected reward from state under fixed policy
Rewards: array of rewards indexing by state, action"
function expectedreward(state::Int, pol::Policy, rewards::Array{Float64, 2})
    a = pol(state)
    rewards[state, a]
end

function expectedreward(state::Int, pol::Policy, getreward::Function)
    getreward(state, pol(state))
end

function expectedreward(states, pol::Policy, reward)
    return collect( expectedreward(s, pol, reward) for s in states)
end

" Performs a single update pass over vector of value estimates."
function updatevalues!(v::Vector{Float64}, gamma::Float64, exprewards::Vector{Float64}, getsuccessors::Function)                                        
    for i in 1:length(v)
        # expected reward when leaving the state
        r = exprewards[i]
        # successor states
        successors, probs = getsuccessors(i)
        v[i] = r + gamma * sum(v[successors] .* probs)
    end
end

"computes action-value matrix"
function actionvalue(values::Vector{Float64}, rewards::Array{Float64, 2}, getsuccessors::Function;gamma=1.0)
    numstate, numaction = size(rewards)
    q = copy(rewards)
    for s in 1:numstate, a in 1:numaction
        sucs, probs = getsuccessors(s, a)
        q[s, a] += gamma * sum(values[sucs] .* probs)
    end
    return q
end

"Evaluates a particular policy"
function policyeval(getsuccessors::Function, exprewards::Vector{Float64}, vals::Vector{Float64};
                        gamma=1.0,tol=1e-4,maxiter=1000)
    prevvals = copy(vals)
    for i in 1:maxiter
        updatevalues!(vals, gamma, exprewards, getsuccessors)
        diff = sum(abs.(vals-prevvals))
        if diff < tol
            @info "Convergence reached, halting after $i iterations"
            break
        end
        prevvals .= vals
    end
    
    return vals
end

policyeval(getsuccessors,exprewards,gamma,tol,maxiter) = policyeval( 
    getsuccessors,exprewards, zeros(length(exprewards)), gamma=gamma,tol=tol,maxiter=maxiter
)

function getsuccessorfactory(envgetsuccessor::Function, pol::Policy)
    return s -> envgetsuccessor(s, pol(s))
end

#make a greedy policy from an action-value function
function makegreedy(actionvalue::Array{Float64, 2})
    numstate = size(actionvalue,1)
    bestactions = reshape(collect(i[2] for i in argmax(actionvalue, dims=2)), numstate)
    return Policy(bestactions)
end


function policyiteration(pol::Policy, getsuccessors::Function, rewards::Array{Float64,2};
                        gamma=1.0, niter=10, neval=3)

    ns, na = size(rewards)
    values = zeros(ns)
    exprewards = expectedreward(1:ns, pol, rewards)
    
    for i in 1:niter
        policysuc = getsuccessorfactory(getsuccessors, pol)
        values = policyeval(policysuc, exprewards, values, gamma=gamma, maxiter=neval)
        actionvalues = actionvalue(values, rewards, getsuccessors, gamma=gamma)
        pol = makegreedy(actionvalues)
        @info("policy: $(pol.actions), values: $values")
    end
    return values, pol
end


function qvalues(state, values::Vector{Float64},
                     getactions::Function,  getreward::Function, getsuccessors::Function; gamma=1.0)

    actions = getactions(state)
    qvals = zeros(length(actions))
    for i in 1:length(actions)
        a = actions[i]
        r = getreward(state, a)
        sucs, probs = getsuccessors(state, a)
        qvals[i] = r + gamma * sum(values[sucs] .* probs)
    end
    return actions, qvals
end

function greedyaction(state, values::Vector{Float64},
                     getactions::Function,  getreward::Function, getsuccessors::Function; gamma=1.0)
    actions, qvals = qvalues(state, values, getactions, getreward, getsuccessors, gamma=gamma)
    return actions[argmax(qvals)], qvals[argmax(qvals)]
end


"perform value interation: returns state-value function"
function valueiteration(statespace, getsuccessors::Function, getactions::Function, getreward::Function ; 
        gamma=1.0, maxiter=100, tol=1e-3)
    
    values = zeros(length(statespace))
    prevvalues = copy(values)
    for i in 1:maxiter
        
        for s in statespace
            a, qmax = greedyaction(s, values, getactions, getreward, getsuccessors, gamma=gamma)
            values[s] = qmax
        end
        
        diff = sum(abs.(values - prevvalues))
        if diff < tol
            @info "tolerance $tol reached after $i steps, halting"
            return values
        end
        prevvalues .= values
    end
    @warn "$maxiter steps without value convergence"
    return values
end
