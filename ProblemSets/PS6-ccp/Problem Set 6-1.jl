using Optim 
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using DataFramesMeta
using CSV 

#We will repeat the estimation of the simplified version of Rust (1987, Econometrica)
# We will exploit the renewal property of the replacement decision and estimate the model using conditional choice probabilities(CCPs)

function wrapall()
    #::::::::::::::::::::::::::::::::::::::::::::::::::
    #Question 1: Read in the data 
    #::::::::::::::::::::::::::::::::::::::::::::::::::

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdata.csv"

    df = CSV.read(HTTP.get(url).body,DataFrame)

    # create bus id variable
    df = @transform(df, :bus_id = 1:size(df,1))

    dfy = @select(df, :bus_id,:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20,:RouteUsage,:Branded)
    dfy_long = DataFrames.stack(dfy, Not([:bus_id,:RouteUsage,:Branded]))
    rename!(dfy_long, :value => :Y)
    dfy_long = @transform(dfy_long, :time = kron(collect([1:20]...),ones(size(df,1))))
    select!(dfy_long, Not(:variable))

    dfx = @select(df, :bus_id,:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20)
    dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
    rename!(dfx_long, :value => :Odometer)
    dfx_long = @transform(dfx_long, :time = kron(collect([1:20]...),ones(size(df,1))))
    select!(dfx_long, Not(:variable))

    df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id,:time])
    sort!(df_long,[:bus_id,:time])

    #::::::::::::::::::::::::::::::::::::::::::::::::::
    #Question 1: Estimate a logit model where the dependent variable is replacement decision 
    #::::::::::::::::::::::::::::::::::::::::::::::::::

    logit_glm = glm(@formula(Y ~ Odometer * Odometer * RouteUsage * RouteUsage * Branded * time * time), df_long, Binomial(), LogitLink()) #logit formula 
    println(logit_glm)

    #fully interacted model where the right hand is a fully interacted set: all terms from 1st to 7th order

    #Dynamic Estimation with CCPs

    #::::::::::::::::::::::::::::::::::::::::::::::::::
    #Question 3: Estimate the thetas
    #::::::::::::::::::::::::::::::::::::::::::::::::::

    #assuming a discount factor beta = 0.9

    #::::::::::::::::::::::::::::::::::::::::::::::::::
    #a: Construct the state transition matrices 
    #::::::::::::::::::::::::::::::::::::::::::::::::::
    include("create_grids.jl")

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdata.csv"
    df = CSV.read(HTTP.get(url).body,DataFrame)
    Y = Matrix(df[:,[:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20]])
    X = Matrix(df[:,[:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20]])
    Z = Vector(df[:,:RouteUsage])
    B = Vector(df[:,:Branded])
    N = size(Y,1)
    T = size(Y,2)
    Xstate = Matrix(df[:,[:Xst1,:Xst2,:Xst3,:Xst4,:Xst5,:Xst6,:Xst7,:Xst8,:Xst9,:Xst10,:Xst11,:Xst12,:Xst13,:Xst14,:Xst15,:Xst16,:Xst17,:Xst18,:Xst19,:Xst20]])
    Zstate = Vector(df[:,:Zst])

    zval,zbin,xval,xbin,xtran = create_grids()

    data_params = (β = 0.9, Y = Y, B = B, N = N, T = T, X = X, Z = Z, Zstate = Zstate, Xstate = Xstate, xtran = xtran, zbin = zbin, xbin = xbin, xval = xval)

    #::::::::::::::::::::::::::::::::::::::::::::::::::
    #b: Compute the future value terms 
    #::::::::::::::::::::::::::::::::::::::::::::::::::

    #create a dataframe that has 4 variables 
    #Oreading = kron(ones(zbin),xval)
    #Rusage = kron(ones(xbin),zval)

    #initialize Branded as 0 and time as 0 
    #Branded = 0 
    #time = 0 
    
    data_state = DataFrame(Odometer= kron(ones(zbin),xval), RouteUsage = kron(ones(xbin),zval), Branded =0, time=0)

    function readdata(data_state, logit_glm, data_params) #write a function that reads in the data frame, the flexible logit estimates and the other state variables
       FV=  zeros(xbin*zbin, 2, T+1) #future value array
       for t=2:T #two nested for loops 
            for b=0:1
                for z=1:zbin
                    for x=1:xbin
                        row = x+(z-1)*xbin
                        temp = DataFrame(Odometer = xval[x], RouteUsage = zval[z], Branded = b, time=t) #updated data frame
                        p1 = predict(logit_glm, temp)
                        p0 =1-p1[1]
                        FV[row, b+1,t]= -β*log(p0)
                    end
                end
            end
        end
            
        for i=1:N
            row0 = ((Zstate)-1)*xbin+1
            for t=1:T
                row1  = (Xstate[i,t] + Zstate[i]-1)*xbin                                                                      # this is the same as row in the first loop, except we use the actual X and Z
                FVT1[i,t]=(xtran[row1,:].-xtran[row0,:])'*FV[row0:row0+xbin-1,B[i]+1,t+1]
            end
            return FVT1'[:]
        end
        
        #::::::::::::::::::::::::::::::::::::::::::::::::::
        #c: estimate the structural parameters
        #::::::::::::::::::::::::::::::::::::::::::::::::::

        #add the output of your future value function as a new column in the original long panel data frame
        df_long = @transform(df_long, FV=FVT1)

        #use GLM package to estimate the structural model; make use of the offset function to add the future value term as another 
        #regressor whose coefficient is restricted to be 1 

        theta_hat_ccp_glm = glm(@formula(Y~Odometer+Branded), df_long, Binomial(), LogitLink(), offset=df_long.FV)
        println(theta_hat_ccp_glm)

    return nothing 
end

#::::::::::::::::::::::::::::::::::::::::::::::::::
#d: I took the 'optional' route
#::::::::::::::::::::::::::::::::::::::::::::::::::

#::::::::::::::::::::::::::::::::::::::::::::::::::
#e: wrap all the code in one function that is called with at time
#::::::::::::::::::::::::::::::::::::::::::::::::::
@time wrapall()