using Random
using LinearAlgebra
using Statistics
using Optim
using DataFrames
using DataFramesMeta
using CSV
using HTTP
using GLM

function allwrap()
    # In this problem set we will explore a simplified version of Rust (1987, Econometrica) bus engine replacement model
    # read in function to create state transitions for dynamic model
    include("create_grids.jl") 

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 1: reshaping the data
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # load in the data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    # create bus id variable
    df = @transform(df, bus_id = 1:size(df,1)) 

    #---------------------------------------------------
    # reshape from wide to long (must do this twice be-
    # cause DataFrames.stack() requires doing it one 
    # variable at a time)
    #---------------------------------------------------
    # first reshape the decision variable
    dfy = @select(df, :bus_id,:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20,:RouteUsage,:Branded)
    dfy_long = DataFrames.stack(dfy, Not([:bus_id,:RouteUsage,:Branded]))
    rename!(dfy_long, :value => :Y)
    dfy_long = @transform(dfy_long, time = kron(collect([1:20]...),ones(size(df,1))))
    select!(dfy_long, Not(:variable))

    # next reshape the odometer variable
    dfx = @select(df, :bus_id,:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20)
    dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
    rename!(dfx_long, :value => :Odometer)
    dfx_long = @transform(dfx_long, time = kron(collect([1:20]...),ones(size(df,1))))
    select!(dfx_long, Not(:variable))

    # join reshaped df's back together
    df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id,:time])
    sort!(df_long,[:bus_id,:time])

    #describe(df_long) #wanted to visualize the data

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 2: estimate a static version of the model
    #:::::::::::::::::::::::::::::::::::::::::::::::::::

    # Estimate Harold Zucher's decision to run buses in his fleet. 
    # x1t  is the mileage on the bus' odometer and b is a dummy variable indicating whether the bus is branded (meaning that the manufacturer is high end)
    # the choice set is 0,1 where 0 denotes replacing the engine 

    form = @formula(Y ~ 1+Odometer+Branded)
    logit = glm(form, df_long, Binomial(), ProbitLink())

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3a: read in data for dynamic model
    #:::::::::::::::::::::::::::::::::::::::::::::::::::

    #Dynamic version of the model using backwards recursion 
    # discount factor beta

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdata.csv"
    df1 = CSV.read(HTTP.get(url).body, DataFrame)

    Y = Matrix(df1[:,[:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20]])
    O = Matrix(df1[:,[:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20]])
    X = Matrix(df1[:,[:Xst1,:Xst2,:Xst3,:Xst4,:Xst5,:Xst6,:Xst7,:Xst8,:Xst9,:Xst10,:Xst11,:Xst12,:Xst13,:Xst14,:Xst15,:Xst16,:Xst17,:Xst18,:Xst19,:Xst20]])
    
    # O for odometer matrix and X for Xst matrix
    #variables :Xst* and :Zst keep track of which discrete bin of the fjs the given observation falls into

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3b: generate state transition matrices
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    zval,zbin,xval,xbin,xtran = create_grids()

    #zval and xval are the grids that correspond to the route usage and odometer reading 
    #zbin and xbin are the number of bins in zval and xval
    #xtran = (zbin*xbin)*xbin Markov transition matrix 

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3c: compute the future value terms
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    #x2 from the formulas is z
    #size(xtran) #number of rows of xtran is 2301
    FV = zeros(20301,2,21) #3 dimensional array of zeroes

    T = 20
    theta = rand(3,1)
    beta =0.9
    #d dummy variable 
    for t=T:-1:1 #t indexes time periods
        for d=0:1 #there is no 0 element of an array, maybe 1 to 2
            for i=1:zbin #looping over the possible permanent route usage states
                for j=1:xbin #looping over the odometer states
                    temp = j+(i-1)*xbin 
                    u1 = theta[1]+theta[2]*xval[j]+theta[3]*d #this part is the uit from the formula in class 
                    EFV1 =xtran[temp,:]'*FV[(i-1)*xbin+1:i*xbin,d+1,t+1]
                    v1 = u1+EFV1
                    u0 =0
                    EFV0 = xtran[1+(i-1)*xbin,:]'*FV[(i-1)*xbin+1:i*xbin,d+1,t+1]
                    v0 =u0+EFV0

                    FV[temp, d+1, t]=beta*log.(exp(v0)+ exp(v1))
                end
            end
        end
    end
    #for d xval[j] = Xstate and d=Zstate

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3d: construct  the log likelihood 
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    
    B=20301 
    for t=1:T #looping over time
        for i=1:B # looping over buses
            loglikelihood = 0 #initializing the loglikelihood as 0 

            row1 = X[i,t]+(Zst[i]-1)*xbin #for the case where the  bus has not been replaced
            row0 = 1 +(Zst[i]-1)*xbin #for the case where the bus has been replaced
            v1 = u1+EFV1 #using the future values from above 
            v0 = u0+EFV0 #u0 is just 0 
            p1 = exp(v1)/exp(v0)+exp(v1)
            p0=1-p1
            
            loglikelihood = (Y[i,t]==1)*log(p1)-(Y[i,t]==0)*log(p0)

        end
    end

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3e: wrap all the code in c) and e) into a function and set it to be passed to Optim
    #:::::::::::::::::::::::::::::::::::::::::::::::::::

    @views @inbounds function myfun(theta) #prepend the macros 

        #:::::::::::::::::::::::::::::::::::::::::::::::::::
        # Question 3c: compute the future value terms
        #:::::::::::::::::::::::::::::::::::::::::::::::::::
        #x2 from the formulas is z
        #size(xtran) #number of rows of xtran is 2301
        FV = zeros(20301,2,21) #3 dimensional array of zeroes

        T = 20
        theta = rand(3,1)
        beta =0.9
        #d dummy variable 
        for t=T:-1:1 #t indexes time periods
            for d=0:1 #there is no 0 element of an array, maybe 1 to 2
                for i=1:zbin #looping over the possible permanent route usage states
                    for j=1:xbin #looping over the odometer states
                        temp = j+(i-1)*xbin 
                        u1 = theta[1]+theta[2]*xval[j]+theta[3]*d #this part is the uit from the formula in class 
                        EFV1 =xtran[temp,:]'*FV[(i-1)*xbin+1:i*xbin,d+1,t+1]
                        v1 = u1+EFV1
                        u0 =0
                        EFV0 = xtran[1+(i-1)*xbin,:]'*FV[(i-1)*xbin+1:i*xbin,d+1,t+1]
                        v0 =u0+EFV0

                        FV[temp, d+1, t]=beta*log.(exp(v0)+ exp(v1))
                    end
                end

            end
        end
    
        #for d xval[j] = Xstate and d=Zstate

        #:::::::::::::::::::::::::::::::::::::::::::::::::::
        # Question 3d: construct  the log likelihood 
        #:::::::::::::::::::::::::::::::::::::::::::::::::::
        
        B=20301 
        for t=1:T #looping over time
            for i=1:B # looping over buses
                loglikelihood = 0 #initializing the loglikelihood as 0 

                row1 = X[i,t]+(Zst[i]-1)*xbin #for the case where the  bus has not been replaced
                row0 = 1 +(Zst[i]-1)*xbin #for the case where the bus has been replaced
                v1 = u1+EFV1 #using the future values from above 
                v0 = u0+EFV0 #u0 is just 0 
                p1 = exp(v1)/exp(v0)+exp(v1)
                p0=1-p1
                
                loglikelihood = (Y[i,t]==1)*log(p1)-(Y[i,t]==0)*log(p0)

            end
        end

        return -loglikelihood
    end

    #loglikelihood_optim = optimize(theta -> myfun(theta), LBFGS(),)Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true
end

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3g: wrap all code in an empty function
#:::::::::::::::::::::::::::::::::::::::::::::::::::
allwrap()



