using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 1: estimate a multinomial logit (with alternative-specific covariates Z)
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#code from assignment

function all()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS3-gev/nlsw88w.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    freqtable(df, :occupation) #Frequency table occupation
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occupation

    #differencing the Z in the likelihood function

    indexedZ = zeros(size(Z,1),size(Z,2))
    for i in 1:size(indexedZ,1)
        indexedZ[i,1] = Z[i,1] - Z[i,8] #professional/technical
        indexedZ[i,2] = Z[i,2] - Z[i,8] #managers/ administrators
        indexedZ[i,3] = Z[i,3] - Z[i,8] #sales
        indexedZ[i,4] = Z[i,4] - Z[i,8] #clerical/unskilled
        indexedZ[i,5] = Z[i,5] - Z[i,8] #craftsmen
        indexedZ[i,6] = Z[i,6] - Z[i,8] #operatives
        indexedZ[i,7] = Z[i,7] - Z[i,8] #transport
        indexedZ[i,7] = Z[i,8] - Z[i,8] #other, make it reference
    end

    function multinomiallogit(coef_vector, X, y)

    # Code from previous assignment
    # Dependent variables in Z
    K = size(X,2)

    # Number of choice variables (occupations)
    J = length(unique(y))

    # Number of observations
    N = length(y)

    gamma_hat = coef_vector[end]
    beta1= coef_vector[1: K * (J-1)]
    beta2 = [reshape(beta1, K, J-1) zeros(K)]

    # This is according to each occupation
    n1 = exp.(X * beta2[:,1] .+ gamma_hat * indexedZ[:,1])
    n2 = exp.(X * beta2[:,2] .+ gamma_hat * indexedZ[:,2])
    n3 = exp.(X * beta2[:,3] .+ gamma_hat * indexedZ[:,3])
    n4 = exp.(X * beta2[:,4] .+ gamma_hat * indexedZ[:,4])
    n5 = exp.(X * beta2[:,5] .+ gamma_hat * indexedZ[:,5])
    n6 = exp.(X * beta2[:,6] .+ gamma_hat * indexedZ[:,6])
    n7 = exp.(X * beta2[:,7] .+ gamma_hat * indexedZ[:,7])
    n8 = exp.(X * beta2[:,8] .+ gamma_hat * indexedZ[:,8])

    denominator = sum(n1 .+ n2 .+  n3.+ n4 .+ n5 .+ n6 .+ n7 .+ n8)

    # Create probability ratios for each choice (occupation)
    p1 = n1 ./ denominator
    p2 = n2 ./ denominator
    p3 = n3 ./ denominator
    p4 = n4 ./ denominator
    p5 = n5 ./ denominator
    p6 = n6 ./ denominator
    p7 = n7 ./ denominator
    p8 = n8 ./ denominator

    log.(p8)
    D = zeros(N,J)
        for j=1:J
            D[:,j] = y.==j
        end
    D

    loglikelihood = sum((D[:,1] .* log.(p1)) .+ (D[:,2] .* log.(p2)) .+ (D[:,3] .* log.(p3))  .+ (D[:,4] .* log.(4))  .+ (D[:,5] .* log.(p5))  .+ (D[:,6] .* log.(p6)) .+ (D[:,7] .* log.(p7)) .+ (D[:,8] .* log.(p8)))
        
    return loglikelihood
    end

    beta_hat_loglikelihood = optimize(coef_vector -> -multinomiallogit(coef_vector, X, y), zeros(22,1), LBFGS(), Optim.Options(g_tol = 1e-6, iterations = 100000, show_trace = true))
    println(beta_hat_loglikelihood.minimizer)

    #[-0.0061003557525688215; 0.17831325390543404; 1.196957428048171; -0.017219210924657686; 0.8894306117340389; 0.2683232888065254; 0.031487917772579095; 0.15256617565105401; 
    #-0.6812969114402306; -10.843416258083902; -0.20267800538080238; -0.06630044627292118; -0.025196567185200594; -0.39972734533749843; 
    #-0.7934853198363228; 0.026245941311256012; -0.9463174080360256; -2.771752628396373; 0.02839807732463528; -0.5125226166101188; -1.504144896766498; 0.3708443786797296;;]

    #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #Question 2: Interpret the estimated coefficient gamma hat
    #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # So above we normalized, and selected a reference category.
    # The estimated coefficient gamma_hat estimates the impact of wages on occupation and tells us how likely an individual is to select one occupation when compared to the reference category. 

    #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3: Nested logit
    #:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    function nestedlogit(coef_vector, X, Z, y)

        # code similar to above
        # Number of dependent variables in X 
        K = size(X,2)

        # Number of occupations
        J = length(unique(y))

        # Number of observations
        N = size(X,1)

        #Initiating the parameters 
        beta_WC = coef_vector[1:3] #beta correspondent to white collar, occupations from 1 to 3
        beta_BC = coef_vector[4:6] #beta correspondent to blue collar, occupations from 4 to 6

        lambda_WC = coef_vector[7]
        lambda_BC = coef_vector[8]
        gamma = coef_vector[9]

        indexedZ = Z .- Z[:,K]

        # Numerators like above but separated by occupation and type of occupation
        n1 = exp.((X * beta_WC .+ gamma * indexedZ[:,1]) ./ lambda_WC) #numerator for 1 and WC
        n2 = exp.((X * beta_WC .+ gamma * indexedZ[:,2]) ./ lambda_WC) #numerator for 2 and WC
        n3 = exp.((X * beta_WC .+ gamma * indexedZ[:,3]) ./ lambda_WC) #numerator for 3 and WC 
        n4 = exp.((X * beta_BC .+ gamma * indexedZ[:,4]) ./ lambda_BC) #numerator for 4 and BC
        n5 = exp.((X * beta_BC .+ gamma * indexedZ[:,5]) ./ lambda_BC) #numerator for 5 and BC
        n6 = exp.((X * beta_BC .+ gamma * indexedZ[:,6]) ./ lambda_BC) #numerator for 6 and BC
        n7 = exp.((X * beta_BC .+ gamma * indexedZ[:,7]) ./ lambda_BC) #numerator for 7 and BC
        n8 = exp.(X * zeros(3) .+ gamma * indexedZ[:,8])

        # Denominator for White collar
        denominator_WC = n1 .+ n2 .+ n3
        println(denominator_WC)

        # Denominator for Blue collar
        denominator_BC = n4 .+ n5 .+ n6 .+ n7
        println(denominator_BC)

        denominator = 1 .+ denominator_WC.^ lambda_WC .+ denominator_BC.^ lambda_BC

        A = zeros(N,J)

        # Iterating through each according to the nest
        for j in 1:J
            if j<=3
                numerator = exp.((X * beta_WC .+ gamma * indexedZ[:,j])./ lambda_WC) .* denominator_WC.^(lambda_WC-1)
                denominator1 = denominator 

                A[:,j] .= numerator ./ denominator1
            end

            if 4 <= j <= 7
                numerator = exp.((X * beta_BC .+ gamma * indexedZ[:,j])./ lambda_BC) .* denominator_BC.^(lambda_BC-1)
                denominator1 = denominator
    
                A[:,j] .= numerator ./ denominator1
            end

            if j==8 
                numerator = n8
                denominator1 = denominator
        
                A[:,j] .= numerator ./ denominator1
        
            end
        end

        # D as above
        D = zeros(N,J)
        for j=1:J
            D[:,j] = y.==j
        end
        D

        loglikelihood = sum(D .* broadcast(log, A))

        return -loglikelihood
    end
end

all()