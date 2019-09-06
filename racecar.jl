import Reinforce: actions, step!, reset!, finished, action, maxsteps
using Reinforce

mutable struct RaceCarEnv <: AbstractEnvironment
    locations::Array{Int, 2}
    velocities::Vector{Int} 
    state::Tuple{Int, Int}
    reward
end

RaceCarEnv(locations::Array{Int, 2}, initstate::Tuple{Int, Int}) = RaceCarEnv(locations, [-1,0, 1], initstate, 0)

"Allowed actions"
function actions(env::RaceCarEnv, s)
    isstart(env, s) && return [(1, 0)]
    actions= []
    for vx in env.velocities
        for vy in env.velocities
            (vx > 0 || vy > 0 ) && push!(actions, (vx, vy))
        end
    end
    return actions
end

ST = Tuple{Int, Int}
SAT = Tuple{ST, ST}

label(env::RaceCarEnv, pos::Tuple{Int, Int}) = env.locations[CartesianIndex(pos)]

function isstart(env::RaceCarEnv, pos::Tuple{Int, Int} )
    return label(env, pos) == -5
end

function isgoal(env::RaceCarEnv, pos::Tuple{Int, Int})
    return label(env, pos) == 5
end

function isout(env::RaceCarEnv, pos::Tuple{Int, Int})
    return label(env, pos) != 0
end

"Implement an environment step"
function step!(env::RaceCarEnv, s::Tuple{Int, Int}, a::Tuple{Int, Int})
    nextpos = s .+ a
    reward = isgoal(env, nextpos) ? 1.0 : 0.0
    env.state = nextpos
    env.reward = reward
    return reward, nextpos
end

function finished(env::RaceCarEnv, s::Tuple{Int, Int})
    return isout(env, s) && ! isstart(env, s)
end

function reset!(env::RaceCarEnv)
    initx = rand(3:6)
    initstate = (1, initx)
    env.state = initstate
    env.reward = 0
end
function RaceCarEnv(locations::Array{Int, 2})
    env = RaceCarEnv(locations, (1,1) )
    reset!(env)
    return env
end

function makeelbow()
    height=10
    width=10
    locations = zeros(Int, height, width)
    locations[1, :] .= -1
    locations[end, :] .= -1
    locations[:, 1] .= -1
    locations[:, end] .= -1
    locations[5:10, end] .= 5
    locations[1, 3:6] .= -5
    return locations
end




function gettrajectory(episode)
    states = []
    actions = []
    rewards = []
    for (s::Tuple{Int, Int}, a::Tuple{Int, Int}, r, __) in episode
        push!(states, s)
        push!(actions, a)
        push!(rewards, r)
    end
    return states, actions, rewards
end

function update!(table::Dict, key, val)
    if haskey(table)
        table[key] += val
    else
        table[key] = val
    end
end

function initQ(env::RaceCarEnv)
    lx, ly = size(env.locations)
    Q = Dict{Tuple{ST, ST}, Float64}()
    for i in 1:lx, j in 1:ly
        for a in actions(env, (i,j))
            Q[((i,j), a)] = randn()
        end
    end
    return Q
end

function initCount(env::RaceCarEnv)
    lx, ly = size(env.locations)
    Q = Dict{Tuple{ST, ST}, Int}()
    for i in 1:lx, j in 1:ly
        for a in actions(env, (i,j))
            Q[((i,j), a)] = 0
        end
    end
    return Q
end

function onpolicyMCcontrolupdate!(Q::Dict{SAT, Float64},C::Dict{SAT, Int}, 
                            states, actions, rewards; gamma=1.0)
    T = length(states)
    #episode accumulated reward
    G = 0
    for t in T:-1:1
        s, a = states[t], actions[t]
        G = rewards[t] + gamma * G
        C[(s,a)] += 1
        Q[(s,a)] += (G - Q[(s,a)]) / C[(s,a)]
    end
    return G
end

struct GreedyPolicy <: AbstractPolicy
    Qtable::Dict{SAT, Float64}
end

action(pol::RandomPolicy, r, s, A) = rand(A)

function action(pol::GreedyPolicy, r, s, A)    
    qvals = collect(pol.Qtable[(s,a)] for a in A)
    return A[argmax(qvals)]
end

ismdp(::RaceCarEnv) = true

function MCpolicyiter!(Q, C, env,niter)
    pol= GreedyPolicy(Q)
    ep = Episode(env, pol)
    for i in 1:niter
        _states, _actions, _rewards = gettrajectory(ep)
        G = onpolicyMCcontrolupdate!(Q, C, _states, _actions, _rewards)
    end
end

maxsteps(env::RaceCarEnv) = 100

env = RaceCarEnv(makeelbow())

Q = initQ(env)
C = initCount(env)

MCpolicyiter!(Q, C,  env, 100)

