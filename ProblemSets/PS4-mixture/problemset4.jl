using Distributions, Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables, ForwardDiff

function allwrap()
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #Question 1: Estimate a multinomial logit (with alternative-specific covariates Z)
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS4-mixture/nlsw88t.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, 
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occ_code

    # Since the occupation categories are the same, code from Solutions to Problem Set 3 
    function multinomiallogit(theta, X, Z, y)
        
        alpha = theta[1:end-1] 
        gamma = theta[end] 

        K = size(X,2)
        J = length(unique(y))
        N = length(y)
        bigY = zeros(N,J)

        for j=1:J
            bigY[:,j] = y.==j
        end
        bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
        
        T = promote_type(eltype(X),eltype(theta))
        num   = zeros(T,N,J)
        dem   = zeros(T,N)
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)
            dem .+= num[:,j]
        end
        
        P = num./repeat(dem,1,J)
        
        loglike = -sum( bigY.*log.(P) )
        
        return loglike
    end

    startvals = [2*rand(7*size(X,2)).-1; .1]
    td = TwiceDifferentiable(theta -> multinomiallogit(theta, X, Z, y), startvals; autodiff = :forward)
    # run the optimizer
    theta_hat_optim_ad = optimize(td, startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    theta_hat_mle_ad = theta_hat_optim_ad.minimizer
    # evaluate the Hessian at the estimates
    H  = Optim.hessian!(td, theta_hat_mle_ad)
    theta_hat_mle_ad_se = sqrt.(diag(inv(H)))
    println("logit estimates with Z")
    println([theta_hat_mle_ad theta_hat_mle_ad_se]) 

    #[0.04037454770423971 0.0028897999619269363; 0.24399335729214336 0.07148911477457666; -1.5713235529089027 0.08750031906722348; 0.04332556311433046 0.002937295380386099; 0.14685474577010746 0.07868303685117227; -2.9591054057919304 0.0927873794176743; 0.1020574976089892 0.0025417619836908645; 0.7473083808095414 0.06752325714893377; -4.120053331815342 0.09010636591997871; 0.037562919493229756 0.003200966334219752; 0.6884898663737034 0.08843481393171432; -3.6557737076695234 0.11140199709519968; 0.020454388006783564 0.0034865834462367317; -0.35840099856738217 0.10795862550139829; -4.3769341550282705 0.1997995170051377; 0.10746374185971029 0.0025994297034883616; -0.526374096868655 0.0739876613110255; -6.199201875407212 0.17809653231057926; 0.11688253567867889 0.002813657271310773; -0.2870558797071566 0.07266358461548288; -5.322251485648969 0.10483945624532444; 1.3074769327085225 0.12466905561776977]

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 2: Does the estimated coefficient gamma make more sense now than in Problem Set 3
    # Answer: Yes, the estimated coefficient gamma makes more sense now then in Problem Set 3, as it appears to measure what it is actually intended to measure. 
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3: re-estimate the mixed logit version of the model in Question 1, with gamma distributed.
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    #Gauss Legendre Quadrature: approximate the integral in 1

    include("lgwt.jl")
    d = Normal(0,1) #mean =0 , std =1 

    #want to verify that integrating over the density of the support equals 1 

    #when using quadrature, pick a number of points and bounds that minimize computing, but offers good approximation of te integral, for normal distributions +/- 4 std. 

    # get quadrature nodes and weights for 7 grid points
    nodes, weights = lgwt(7,-4,4) 

    #compute the interal and verify it is 1
    sum(weights.*pdf.(d,nodes))

    #compute expectation and verify it's 0
    sum(weights.*nodes.*pdf.(d,nodes))

    #More practice 

    d1= Normal(0,2)
    nodes1, weights1 = lgwt(7,-5,5)
    sum(weights1.*pdf.(d1, nodes1))

    nodes2, weights2 =lgwt(10,-5,5)
    sum(weights2.*pdf.(d1, nodes2))

    #The quadrature approximates the true value relatively well, computing approximates that are approximately the same in value. 

    #Monte Carlo Integration
    D=rand(1_000_000)
    b=5
    a=-5
    N = Normal(0,2) # mean = 0, standard deviation = 2
    nodesMC, weightsMC = lgwt(D,(b-a)/D, (b-a)/D)
    sum(weightsMC.*pdf.(N, nodesMC))
    
    D1= rand(1_000)
    nodesMC1, weightsMC1 = lgwt(D1,(b-a)/D1, (b-a)/D1)
    sum(weightsMC1.*pdf.(N, nodesMC1))

    # The simulated integral approximates the true value when D is higher, therefore 1000000, not when D is smaller. 
    
    # d) Note the similarity between quadrature and Monte Carlo - the quadrature weight in 
    # Monte Carlo integration is the same (b-a)/D at each node, and the quadrature node is U[a,b] random number

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 4: try to modify your code from question 1 to optimize the likelihood function in 2
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    #function multinomiallogit1(theta, X, Z, y)
        
        #alpha = theta[1:end-1] 
        #gamma = theta[end] 

        #K = size(X,2)
        #J = length(unique(y))
        # N = length(y)
        #bigY = zeros(N,J)

        #for j=1:J
            #bigY[:,j] = y.==j
        #end
        #bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
        
        #T = promote_type(eltype(X),eltype(theta))
        #num   = zeros(T,N,J)
        #dem   = zeros(T,N)
        #for j=1:J
            #num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)
            #dem .+= num[:,j]
        #end
        
        #P = num./repeat(dem,1,J)
        
        #loglike = -sum( bigY.*log.(P) )
        
        #return loglike
    #end

    #params = [0.04037454770423971 0.0028897999619269363; 0.24399335729214336 0.07148911477457666; -1.5713235529089027 0.08750031906722348; 0.04332556311433046 0.002937295380386099; 0.14685474577010746 0.07868303685117227; -2.9591054057919304 0.0927873794176743; 0.1020574976089892 0.0025417619836908645; 0.7473083808095414 0.06752325714893377; -4.120053331815342 0.09010636591997871; 0.037562919493229756 0.003200966334219752; 0.6884898663737034 0.08843481393171432; -3.6557737076695234 0.11140199709519968; 0.020454388006783564 0.0034865834462367317; -0.35840099856738217 0.10795862550139829; -4.3769341550282705 0.1997995170051377; 0.10746374185971029 0.0025994297034883616; -0.526374096868655 0.0739876613110255; -6.199201875407212 0.17809653231057926; 0.11688253567867889 0.002813657271310773; -0.2870558797071566 0.07266358461548288; -5.322251485648969 0.10483945624532444; 1.3074769327085225 0.12466905561776977]
    #td1 = TwiceDifferentiable(theta -> multinomiallogit1(theta, X, Z, y), params; autodiff = :forward)
    # run the optimizer
    #theta_hat_optim= optimize(td1, params, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
    #theta_hat_ad = theta_hat_optim.minimizer
    # evaluate the Hessian at the estimates
    #H  = Optim.hessian!(td1, theta_hat_ad)
    #theta_hat_ad_se = sqrt.(diag(inv(H)))
    #println([theta_hat_ad theta_hat_ad_se]) 

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #Question 5: modify code from question 1 to optimize the likelihood function, where the integral is approximated by Monte Carlo 
    #The program will take the same form as under quadrature , but the weights will be slightly different 
    #instead of 7 quadrature points, use many, many simulation draws.

    #Monte Carlo Integration Part 
    #function multinomiallogitMC(theta, X, Z, y)
        
        #K = size(X,2)
        #J = length(unique(y))
        #N = length(y)
        #bigY = zeros(N,J)

        #alpha = theta[1:end-1] 
        #gamma = theta[end] 

        #D=rand(1_000_000)
        #N = Normal(0,2) # mean = 0, standard deviation = 2
        #nodesMC, weightsMC = lgwt(D,(b-a)/D, (b-a)/D)
        #sum(weightsMC.*pdf.(N, nodesMC))

        #for j=1:J
            #bigY[:,j] = y.==j
        #end
        #bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
        
        #T = promote_type(eltype(X),eltype(theta))
        #num   = zeros(T,N,J)
        #dem   = zeros(T,N)
        #for j=1:J
            #num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)
            #dem .+= num[:,j]
       #end
        
       # P = num./repeat(dem,1,J)
        
        #loglike = -sum( bigY.*log.(P) )
        
       # return loglikeMC
    end
end

allwrap()

#I know this is not correct, but honestly I was afraid to look anything up after your email on Friday. 
# And not being able to run it,made it harder for me for sure. 
