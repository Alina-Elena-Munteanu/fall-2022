using SMM
using Optim
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using DataFramesMeta
using CSV
using FreqTables
using ForwardDiff
# models by Generalized Method of Moments (GMM) and Simulated Method of Moments (SMM)

function wrapall()
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 1: estimate the linear regression from q2 PS2 by GMM
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.married.==1

    function ols_gmm(beta, X, y)
        K = size(X,2)
        N = size(y,1)
        P= X*beta
        g = y .- P
        J = g'*I*g
    return J
    end

    beta_optim = optimize(beta -> ols_gmm(beta, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
    println(beta_optim.minimizer)

    #hint: question 2: bigY[:] .- P[:]
    #[0.6613509513050156, -0.004625908038887945, 0.2259504322996523, -0.012184197712742177]
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 2: Estimate the multinomial logit model from Question 5 of Problem Set 2
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::

    #Maximum Likelihood: solutions code 
    freqtable(df, :occupation) # note small number of obs in some occupations
    df = dropmissing(df, :occupation)
    df[df.occupation.==8 ,:occupation] .= 7
    df[df.occupation.==9 ,:occupation] .= 7
    df[df.occupation.==10,:occupation] .= 7
    df[df.occupation.==11,:occupation] .= 7
    df[df.occupation.==12,:occupation] .= 7
    df[df.occupation.==13,:occupation] .= 7
    freqtable(df, :occupation) # problem solved

    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.occupation

    function mlogit(alpha, X, y)
            
        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
            
        num = zeros(N,J)
        dem = zeros(N)
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j])
            dem .+= num[:,j]
        end
            
        P = num./repeat(dem,1,J)
            
        loglike = -sum( bigY.*log.(P) )
            
        return loglike
    end

    alpha_zero = zeros(6*size(X,2))
    alpha_rand = rand(6*size(X,2))
    alpha_true = [.1910213,-.0335262,.5963968,.4165052,-.1698368,-.0359784,1.30684,-.430997,.6894727,-.0104578,.5231634,-1.492475,-2.26748,-.0053001,1.391402,-.9849661,-1.398468,-.0142969,-.0176531,-1.495123,.2454891,-.0067267,-.5382892,-3.78975]
    alpha_start = alpha_true.*rand(size(alpha_true))
    println(size(alpha_true))
    alpha_hat_optim = optimize(a -> mlogit(a, X, y), alpha_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    alpha_hat_mle = alpha_hat_optim.minimizer
    println(alpha_hat_mle)
    

    #::::::::::::::::::::::::::::::::::::::::::::::::::
    #b: GMM with the MLE estimates as starting values
    #:::::::::::::::::::::::::::::::::::::::::::::::::: 
    freqtable(df, :occupation) # note small number of obs in some occupations
    df = dropmissing(df, :occupation)
    df[df.occupation.==8 ,:occupation] .= 7
    df[df.occupation.==9 ,:occupation] .= 7
    df[df.occupation.==10,:occupation] .= 7
    df[df.occupation.==11,:occupation] .= 7
    df[df.occupation.==12,:occupation] .= 7
    df[df.occupation.==13,:occupation] .= 7
    freqtable(df, :occupation) # problem solved

    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.occupation

    function mlogit2(alpha, X, y)
            
        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
        bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
            
        num = zeros(N,J)
        dem = zeros(N)
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j])
            dem .+= num[:,j]
        end
        
        P = num./repeat(dem,1,J)
        g = bigY[:].-P[:]
        J = g'*I*g
            
        return J
    end

    #startvals=[0.1909911054784581, -0.03352543877634775, 0.5963969417984843, 0.4165039662342874, -0.16986562838197156, -0.03597756918925885, 1.3068362779913967, -0.4309975119480089, 0.6894579706732944, -0.01045744730510346, 0.5231625314466168, -1.4924750971913858, -2.267524300853579, -0.005299001757111118, 1.3914041121018026, -0.9849648114859104, -1.398492575800385, -0.014296196723452191, -0.017659164458536616, -1.4951341962316702, 0.2454672199571531, -0.006726137769123928, -0.5382903217977227, -3.789791563826617]
    #td = TwiceDifferentiable(alpha -> mlogit2(alpha, X, y), startvals; autodiff = :forward)
    
    #mlogit2_optim = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    #mlogit2_final =  mlogit2_optim.minimizer
    #println(mlogit2_final)

    #the g object should be a vector of dimension NxJ where N is the number of rows of the X matrix and J is the dimension of the choice set 
    # each element g = d-P where d and P are stacked vectors of dimenson NxJ

    #MLE Estimates from above

    #::::::::::::::::::::::::::::::::::::::::::::::::::
    #c: GMM with random starting values 
    #:::::::::::::::::::::::::::::::::::::::::::::::::: 

    # Compare your estimates from b) and c). Is the objective function globally concave?
    # Answer: No, the objective function is not globally concave. It completely depends on the starting point on the curve. 

    #::::::::::::::::::::::::::::::::::::::::::::::::::
    #3: Simulate a data set from a multinomial logit model
    #:::::::::::::::::::::::::::::::::::::::::::::::::: 

    function my_logit(N,K,J)
        X=rand(N,K) #generate X using a random number generator 
        beta = hcat(rand(K,J-1), zeros(K))
        P=exp(X*beta)./sum(eachrow(X*beta))
        ϵ = rand(N)
        
        y=zeros(N)
        for i=1:N
            if ϵ[i]>=0 && ϵ[i]<=P[i,1]
                y[i]=1
            elseif P[i,1]<ϵ[i] && ϵ[i]<=P[i,2]
                y[i]=2
            elseif P[i,2]<ϵ[i] && ϵ[i]<=P[i,3]
                y[i]=3
            elseif P[i,J-1]<ϵ[i] && ϵ[i]<=1
                y[i]=J
            end
        end

        return X, y, beta
    end

    #::::::::::::::::::::::::::::::::::::::::::::::::::
    #5: Use the code from question 3 to estimate the multinomial logit model from question 2 using SMM 
    #:::::::::::::::::::::::::::::::::::::::::::::::::: 

    #Slide 18 from Lecture 9 slide deck
    #function ols_smm(θ, X, y, D)
        #K = size(X,2)
        #N = size(y,1)
        #β = θ[1:end-1]
        #σ = θ[end]
        #if length(β)==1
            #β = β[1]
        #end
        # N+1 moments in both model and data
        #gmodel = zeros(N+1,D)
    #end

    function mlogit_SMM(alpha, X, y, D)
        K = size(X,2)
        J = length(unique(y))
        N = size(y,1)
        bigY = zeros(N,J)
        
        y = zeros(N,D)

        for j=1:J
            bigY[:,j] = y.==j
        end
        bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
        
        gmodel = zeros(N,D)
        gdata = vcat(y)

        P=exp.(X*bigAlpha)./sum.(eachrow(X*bigAlpha))
        for d=1:D
            ε = randn(N)
            for i=1:N
                if ε[i]>=0 && ε[i]<P[i,1]
                    ỹ[i,d]=1 
                elseif P[i,1]<ϵ[i] && ϵ[i]<=P[i,2]
                    ỹ[i,d]=2
                elseif P[i,2]<ϵ[i] && ϵ[i]<=P[i,3]
                    ỹ[i,d]=3
                elseif P[i,3]<ϵ[i] && ϵ[i]<=P[i,4]
                    ỹ[i,d]=4
                elseif P[i,4]<ϵ[i] && ϵ[i]<=P[i,5]
                    ỹ[i,d]=5
                elseif P[i,5]<ϵ[i] && ϵ[i]<=P[i,6]
                    ỹ[i,d]=6
                elseif P[i,J-1]<ϵ[i] && ϵ[i]<=1 
                    ỹ[i,d]=J #going up to occupations
                end
            end

            gmodel[:,d] = ỹ
        end

        g= vec(gdata .- mean(gmodel; dims=2))
        
        J = g'*I*g
    
        return J
    end
end 

#Question 6
wrapall()
    