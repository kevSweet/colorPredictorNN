

library(sigmoid)
library(png)

# 0 -> Black 1 -> White. Predefined function for contrasting text with background
yiqCalc <- function(rgbmatrix){
  
  r <- rgbmatrix[1,1,1] * 255 
  g <- rgbmatrix[1,1,2] * 255
  b <- rgbmatrix[1,1,3] * 255
  
  yiq <- (r*299 + g*587 + b*114)/1000
  
  return(if(yiq >= 128) 0 else 1)
  
}


#generates 10x10 random pixel images of a single color
imageGenerator <- function(numofImages){
  l <- list()
  for(i in 1:numofImages){
    image <- array(NA, c(10,10,3))
    image[,,1] <- runif(1,min = 0, max = 1)
    image[,,2] <- runif(1,min = 0, max = 1)
    image[,,3] <- runif(1,min = 0, max = 1)
    l[[i]] <- image
  }
  
  return(l)
  
}


viewImages <- function(completedPrediction, yiqCalc){
  
}

prdictNN <- function(wghts,b,clrdimg){
    z <- clrdimg[1,1,1]*wghts[1]+clrdimg[1,1,2]*wghts[2]+clrdimg[1,1,3]*wghts[3]+b
    return(sigmoid(z))
}

main <- function(){
  #generate test/train images
  ltrain <- imageGenerator(10000)
  ltest <- imageGenerator(2000)
  
  #generate output of yiq function (function we will be training NN to) for train and test set
  lyiqtrain <- array(NA,dim = length(ltrain))
  lyiqtest <- array(NA,dim = length(ltest))
  for(i in 1:length(ltrain)){
    lyiqtrain[i] <- yiqCalc(ltrain[[i]])
    if(i<=2000){
      lyiqtest[i] <- yiqCalc(ltest[[i]])
    }
  }
  
  #initialize neural network weights and bias
  wghts <- rnorm(3)
  b <- rnorm(1)
  learn_rate <- 0.2
  
  #train
  for(i in 1:50000){
    r_indx <- runif(1,min = 1,max = 10000)
    selPoint <- ltrain[[r_indx]]
    target <- lyiqtrain[[r_indx]]
    
    #feed forward
    pred <- prdictNN(wghts,b,selPoint)
    
    cost <- (pred - target)^2
    
    dcost_dpred <- 2 * (pred - target)
    
    #d/dz sigmoid z = sigmoid(z)*(1-sigmoid(z))
    dpred_dz <- sigmoid(z) * (1-sigmoid(z))
    
    #how does z change with respect to the weights and b?
    dz_dw1 <- selPoint[1,1,1]
    dz_dw2 <- selPoint[1,1,2]
    dz_dw3 <- selPoint[1,1,3]
    dz_db <- 1
    
    #now we can get the partial derivatives using the chain rule
    #We are bringing how the cost changes through each function, first through the square, then through the sigmoid
    #and finally whatever is multiplying our parameter of interest becomes the last past
    dcost_dw1 <- dcost_dpred * dpred_dz * dz_dw1
    dcost_dw2 <- dcost_dpred * dpred_dz * dz_dw2
    dcost_dw3 <- dcost_dpred * dpred_dz * dz_dw3
    dcost_db <- dcost_dpred * dpred_dz * dz_db
    
    
    #update parameters
    wghts[1] <- wghts[1] - learn_rate*dcost_dw1
    wghts[2] <- wghts[2] - learn_rate*dcost_dw2
    wghts[3] <- wghts[3] - learn_rate*dcost_dw3
    b <- b - learn_rate*dcost_db
  }

  
  #test
  for(i in 1:2000){
    selPoint <- ltest[[i]]
    target <- lyiqtest[[i]]
    pred <- prdictNN(wghts,b,selPoint)
    cost <- (pred - target)^2
  }
  
}


