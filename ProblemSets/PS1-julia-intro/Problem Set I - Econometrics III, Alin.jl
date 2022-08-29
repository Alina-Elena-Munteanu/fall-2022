# Problem Set I - Econometrics III, Alina-Elena Munteanu (23rd of August 2022)

#Before Starting
#add JLD2
#add Randomdd
#add LinearAlgebra
#add Statistics
#add CSV
#add DataFrames
#add FreqTables 
#add Distributions
#add Kronecker

###########################################################################
# Item 1: Initializing Variables & Matrix Operations
###########################################################################

using Random, Distributions, JLD2, CSV, DataFrames, LinearAlgebra, Statistics, FreqTables, Kronecker
Random.seed!(1234)

function q1()
    #Generating matrix A, uniformly distributed
    A=rand(Uniform(-5,10),10,7)

    #Generating matrix B, normally distributed
    B=rand(Normal(-2,15),10,7)

    #Generating matrix C, horizontal concatenation
    C=cat(A[1:5, 1:5],B[1:5, 6:7]; dims=2)
    
    #Generating matrix D
    D=A.*(A.<=0)

    #List in all elements in A
    println(length(A))
    #or option#2
    #A[:]

    #Unique elements in matrix D
    println(length(unique(D)))

    #Create matrix E
    E=reshape(B,length(B))
    #E=vec(B)
    #E=B[:]

    #Create a new array F which is 3-dimensional and contains A in the first column of the third dimension and B in the second column of the third dimension
    F=cat(A,B, dims=3)

    #Use the permutedims() functions to twist F to 2x10x7, save the new matrix as F
    F=permutedims(F,[3 1 2])

    #Create matrix G which equals the Kronecker product of B and C
    G=kronecker(B,C)

    #What happens when you try kronecker(C,F)
    #Answer: When I attempt to compute the Kronecker product for matrices C and F, an error occurs. The error stems from the dimensions of the matrices being different due to the permutation of F above. 

    #Save the matrices A,B,C,D,E,F and G as a jld file named matrixpractice

    JLD2.save("matrixpractice.jld","matrix A", A,"matrix B",B,"matrix C",C,"matrix D",D,"matrix E",E,"matrix F",F,"matrix G",G)
    #Checking it saved the correct output
    #JLD2.load("matrixpractice.jld")

    #Save only matrices A,B,C,D as jld files called firstmatrix
    save("firstmatrix.jld","matrix A", A,"matrix B",B,"matrix C",C,"matrix D",D)

    #Export C as a CSV file, transform C into a DataFrame first
   
    #CSV.write("Cmatrix.csv", DataFrame(C))
   
    #Or maybe
    C_df= DataFrame(C, :auto)
    CSV.write("Cmatrix.csv", C_df)

    #Export D as a tab-delimited file called Dmatrix
    D_df=DataFrame(D,:auto)
    CSV.write("Dmatrix.csv", D_df; delim="\t")

    return A, B, C, D
end

#Function question 1, output arrays A, B, C, D

##############################################################################
#Item 2: Practice with Loops and Comprehensions
##############################################################################

function q2(A,B,C)
    #Product of A and B element by elememt with loop or Comprehension

    row=size(A,1)
    col=size(A,2)
    AB=zeros(row,col)

    for i=1:row j=1:col
        AB[i,j]=A[i,j]*B[i,j]
    end

    #Method2 - Loop 
    #function q2(A,B,C)
    #AB=[A[i,j]*B[i,j] for i in 1:10, j in 1:7]

    #Product of A and B element by element with no loop or comprehension
    AB2 = A.*B

    #Creating a column vector Cprime which contains the elements of C that are in between -5 and 5
    Cprime =  [C[i,j] for i in 1:5, j in 1:7 if -5 <= C[i,j] <= 5]
    # Cprime2 without a loop 
    Cprime2=c[(C.>=-5) .& (C<=5)]
    #Or maybe
    #Cprime2 = C[-5 .<= C .<=5]

    #Using loops or comprehensions create a 3 dimensional array called X that is of dimensions NxKxT
    N, K, T = 15_169, 6, 5 
    X= cat([cat([ones(N,1) rand(N,1).<=(0.75*(6-t)/5) (15+t-1).+(5*(t-1)).*randn(N,1) (π*(6-t)/3).+(1/exp(1)).*randn(N,1) rand(Binomial(20,0.6),N) rand(Binomial(20,0.5),N)];dims=3) for t=1:T]...;dims=3) 

    # Use comprehensives to create a matrix beta which is KxT and whose elements evolve across time
    β=zeros(K,T);
    β[1,:]=[1:0.25:2;];
    β[2,:]=[log(j) for j in 1:T];
    β[3,:]=[-sqrt(j) for j in 1:T];
    β[4,:]=[exp(j) - exp(j+1) for j in 1:T];
    β[5,:]=[j for j in 1:T];
    β[6,:]=[j/3 for j in 1:T];
    β
    #Use comprehensions to create a matrix Y which is NxT defined by 
    Y = hcat([cat(X[:,:,t]*β[:,t] + .36*randn(N,1);dims=2) for t=2:T])
    return nothing
end

#############################################################################
#Item3: Reading in Data and calculating summary statistics 
#############################################################################

function q3()
    #Import the file nlsw88.csv into Julia as a data frame, appropriately convert missing values and variable names
    data=CSV.read("/Users/alinaelena/Desktop/PhD Semester III/Ransom III/ProblemSets/PS1-julia-intro", DataFrame,header=true,missingstring="")
    save("nlsw88.jld","nlsw88", data)

    #What percentage of the sample has never been married? What percentage are college graduates 
    describe(data)
    mean(nlsw.never_married)
    mean(nlsw88.collgrad)

    # The never married mean is 0.104185
    # There are 2 observations missing.

    #Use the freqtable() to report what percentage of the sample is in each race category 
    freqtable(nlsw88, :race)

    #Use the describe() to create a matrix called summarystats which lists the mean, median, standard deviation, min, max, number of unique elements, and interquartile range(75h percentile minus 25th percentile) of the dataframe. How many grade observations are missing?
    summarystats=describe(nlsw88)

    #Show the joint distribution of industry and occupation using cross-tabulation
    freqtable(nlsw88, :industry, :occupation)

    #Tabulate the mean wage over industry and occupation categories
    mean_wage= nlsw88[:,[:industry, :occupation, :wage]]
    how_to_group = groupby(mean_wage,[:industry,:occupation])
    combine(how_to_group, valuecols(how_to_group).=>mean)

    return nothing
end

#Wrap a function around all the code for question 3 

function q4()
    matrices = load("firstmatrix.jld")
    A = matrices["A"]
    B = matrices["B"]
    C = matrices["C"]
    D = matrices["D"]

    function matrixops(matrix1,matrix2)

        #matrixops has three outputs computing the following: output1 = the element by element product, output2 = the product of matrix A transposed and B, output3 = the sum of all elements in A and b

        if size(matrix1) != size(matrix2)
            error("Inputs must have the same size.")
        end

        output1=matrix1.*matrix2
        output2=matrix1'*matrix2
        output3=sum(matrix1+matrix2)

        return output1, output2, output3
    end

    matrixops(A,B)

    #matrixops(C,D)

    #matrix1=load("nlsw88.jld")
    #nlsw88=matrix1["nlsw88"]
    #matrixops(convert(Array,nlsw88.ttl_exp),convert(Array,nlsw88.wage))

    return nothing
end

A,B,C,D=q1()
q2(A,B,C)
q3()
q4()



