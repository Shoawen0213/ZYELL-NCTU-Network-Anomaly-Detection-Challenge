import numpy as np
import math
import random

Alpha = 0.3
Beta  = 2
Normal     = 1
DOS        = 0
Ip_sweep   = 2
Nmap       = 3
Port_sweep = 4

def score(Predicted_label,Real_label):
    data_count = np.size(Predicted_label)
    TN_Normal = 0
    TP_Normal = 0
    FN_Normal = 0
    FP_Normal = 0
    TN_DOS = 0
    TP_DOS = 0
    FN_DOS = 0
    FP_DOS = 0
    TN_IP_sweep = 0
    TP_IP_sweep = 0
    FN_IP_sweep = 0
    FP_IP_sweep = 0
    TN_Nmap = 0
    TP_Nmap = 0
    FN_Nmap = 0
    FP_Nmap = 0
    TN_Port_sweep = 0
    TP_Port_sweep = 0
    FN_Port_sweep = 0
    FP_Port_sweep = 0
    total_cost = 0
    max_cost = 0
    for i in range(data_count):
        if(Real_label[i] == Normal):
            if(Predicted_label[i] == Normal):
                TP_Normal += 1
                TN_DOS += 1
                TN_IP_sweep += 1
                TN_Nmap += 1
                TN_Port_sweep += 1
            elif(Predicted_label[i] == DOS):
                FN_Normal += 1
                FP_DOS += 1
                TN_IP_sweep += 1
                TN_Nmap += 1
                TN_Port_sweep += 1
                total_cost += 2
                if(max_cost<2):
                    max_cost = 2
            elif(Predicted_label[i] == Ip_sweep):
                FN_Normal += 1
                TN_DOS += 1
                FP_IP_sweep += 1
                TN_Nmap += 1
                TN_Port_sweep += 1
                total_cost += 1
                if(max_cost<1):
                    max_cost = 1
            elif(Predicted_label[i] == Nmap):
                FN_Normal += 1
                TN_DOS += 1
                TN_IP_sweep += 1
                FP_Nmap += 1
                TN_Port_sweep += 1
                total_cost += 1
                if(max_cost<1):
                    max_cost = 1
            elif(Predicted_label[i] == Port_sweep):
                FN_Normal += 1
                TN_DOS += 1
                TN_IP_sweep += 1
                TN_Nmap += 1
                FP_Port_sweep += 1
                total_cost += 1
                if(max_cost<1):
                    max_cost = 1
        elif(Real_label[i] == DOS):
            if(Predicted_label[i] == DOS):
                TN_Normal += 1
                TP_DOS += 1
                TN_IP_sweep += 1
                TN_Nmap += 1
                TN_Port_sweep += 1
            elif(Predicted_label[i] == Normal):
                FP_Normal += 1
                FN_DOS += 1
                TN_IP_sweep += 1
                TN_Nmap += 1
                TN_Port_sweep += 1
                total_cost += 2
                if(max_cost<2):
                    max_cost = 2
            elif(Predicted_label[i] == Ip_sweep):
                TN_Normal += 1
                FN_DOS += 1
                FP_IP_sweep += 1
                TN_Nmap += 1
                TN_Port_sweep += 1
                total_cost += 2
                if(max_cost<2):
                    max_cost = 2
            elif(Predicted_label[i] == Nmap):
                TN_Normal += 1
                FN_DOS += 1
                TN_IP_sweep += 1
                FP_Nmap += 1
                TN_Port_sweep += 1
                total_cost += 2
                if(max_cost<2):
                    max_cost = 2
            elif(Predicted_label[i] == Port_sweep):
                TN_Normal += 1
                FN_DOS += 1
                TN_IP_sweep += 1
                TN_Nmap += 1
                FP_Port_sweep += 1
                total_cost += 2
                if(max_cost<2):
                    max_cost = 2
        elif(Real_label[i] == Ip_sweep):
            if(Predicted_label[i] == Ip_sweep):
                TN_Normal += 1
                TN_DOS += 1
                TP_IP_sweep += 1
                TN_Nmap += 1
                TN_Port_sweep += 1
            elif(Predicted_label[i] == Normal):
                FP_Normal += 1
                TN_DOS += 1
                FN_IP_sweep += 1
                TN_Nmap += 1
                TN_Port_sweep += 1
                total_cost += 1
                if(max_cost<1):
                    max_cost = 1
            elif(Predicted_label[i] == DOS):
                TN_Normal += 1
                FP_DOS += 1
                FN_IP_sweep += 1
                TN_Nmap += 1
                TN_Port_sweep += 1
                total_cost += 1
                if(max_cost<1):
                    max_cost = 1
            elif(Predicted_label[i] == Nmap):
                TN_Normal += 1
                TN_DOS += 1
                FN_IP_sweep += 1
                FP_Nmap += 1
                TN_Port_sweep += 1
                total_cost += 1
                if(max_cost<1):
                    max_cost = 1
            elif(Predicted_label[i] == Port_sweep):
                TN_Normal += 1
                TN_DOS += 1
                FN_IP_sweep += 1
                TN_Nmap += 1
                FP_Port_sweep += 1
                total_cost += 1
                if(max_cost<1):
                    max_cost = 1
        elif(Real_label[i] == Nmap):
            if(Predicted_label[i] == Nmap):
                TN_Normal += 1
                TN_DOS += 1
                TN_IP_sweep += 1
                TP_Nmap += 1
                TN_Port_sweep += 1
            elif(Predicted_label[i] == Normal):
                FP_Normal += 1
                TN_DOS += 1
                TN_IP_sweep += 1
                FN_Nmap += 1
                TN_Port_sweep += 1
                total_cost += 1
                if(max_cost<1):
                    max_cost = 1
            elif(Predicted_label[i] == DOS):
                TN_Normal += 1
                FP_DOS += 1
                TN_IP_sweep += 1
                FN_Nmap += 1
                TN_Port_sweep += 1
                total_cost += 1
                if(max_cost<1):
                    max_cost = 1
            elif(Predicted_label[i] == Ip_sweep):
                TN_Normal += 1
                TN_DOS += 1
                FP_IP_sweep += 1
                FN_Nmap += 1
                TN_Port_sweep += 1
                total_cost += 1
                if(max_cost<1):
                    max_cost = 1
            elif(Predicted_label[i] == Port_sweep):
                TN_Normal += 1
                TN_DOS += 1
                TN_IP_sweep += 1
                FN_Nmap += 1
                FP_Port_sweep += 1
                total_cost += 1
                if(max_cost<1):
                    max_cost = 1
        elif(Real_label[i] == Port_sweep):
            if(Predicted_label[i] == Port_sweep):
                TN_Normal += 1
                TN_DOS += 1
                TN_IP_sweep += 1
                TN_Nmap += 1
                TP_Port_sweep += 1
            elif(Predicted_label[i] == Normal):
                FP_Normal += 1
                TN_DOS += 1
                TN_IP_sweep += 1
                TN_Nmap += 1
                FN_Port_sweep += 1
                total_cost += 1
                if(max_cost<1):
                    max_cost = 1
            elif(Predicted_label[i] == DOS):
                TN_Normal += 1
                FP_DOS += 1
                TN_IP_sweep += 1
                TN_Nmap += 1
                FN_Port_sweep += 1
                total_cost += 1
                if(max_cost<1):
                    max_cost = 1
            elif(Predicted_label[i] == Ip_sweep):
                TN_Normal += 1
                TN_DOS += 1
                FP_IP_sweep += 1
                TN_Nmap += 1
                FN_Port_sweep += 1
                total_cost += 1
                if(max_cost<1):
                    max_cost = 1
            elif(Predicted_label[i] == Nmap):
                TN_Normal += 1
                TN_DOS += 1
                TN_IP_sweep += 1
                FP_Nmap += 1
                FN_Port_sweep += 1
                total_cost += 1
                if(max_cost<1):
                    max_cost = 1
    max_cost = max_cost * data_count
    if(TP_Normal+FP_Normal==0):
        Precision_Normal = 0
    else:
        Precision_Normal = TP_Normal/(TP_Normal+FP_Normal)
    #Accuracy_Normal  = (TP_Normal+TN_Normal)/(TP_Normal+FP_Normal+FN_Normal+TN_Normal)
    if(TP_Normal+FN_Normal==0):
        Recall_Normal = 0
    else:
        Recall_Normal = TP_Normal/(TP_Normal+FN_Normal)
    if(TP_DOS+FP_DOS==0):
        Precision_DOS = 0
    else:
        Precision_DOS = TP_DOS/(TP_DOS+FP_DOS)
    #Accuracy_DOS  = (TP_DOS+TN_DOS)/(TP_DOS+FP_DOS+FN_DOS+TN_DOS)
    if(TP_DOS+FN_DOS==0):
        Recall_DOS = 0
    else:
        Recall_DOS = TP_DOS/(TP_DOS+FN_DOS)
    if(TP_IP_sweep+FP_IP_sweep==0):
        Precision_IP_sweep = 0
    else:    
        Precision_IP_sweep = TP_IP_sweep/(TP_IP_sweep+FP_IP_sweep)
    #Accuracy_IP_sweep  = (TP_IP_sweep+TN_IP_sweep)/(TP_IP_sweep+FP_IP_sweep+FN_IP_sweep+TN_IP_sweep)
    if(TP_IP_sweep+FN_IP_sweep==0):
        Recall_IP_sweep = 0
    else:    
        Recall_IP_sweep    = TP_IP_sweep/(TP_IP_sweep+FN_IP_sweep)
    if(TP_Nmap+FP_Nmap==0):
        Precision_Nmap = 0
    else:    
        Precision_Nmap = TP_Nmap/(TP_Nmap+FP_Nmap)
    #Accuracy_Nmap  = (TP_Nmap+TN_Nmap)/(TP_Nmap+FP_Nmap+FN_Nmap+TN_Nmap)
    if(TP_Nmap+FN_Nmap==0):
        Recall_Nmap = 0
    else:
        Recall_Nmap    = TP_Nmap/(TP_Nmap+FN_Nmap)
    if(TP_Port_sweep+FP_Port_sweep==0):
        Precision_Port_sweep = 0
    else:    
        Precision_Port_sweep = TP_Port_sweep/(TP_Port_sweep+FP_Port_sweep)
    #Accuracy_Port_sweep  = (TP_Port_sweep+TN_Port_sweep)/(TP_Port_sweep+FP_Port_sweep+FN_Port_sweep+TN_Port_sweep)
    if(TP_Port_sweep+FN_Port_sweep==0):
        Recall_Port_sweep = 0
    else:
        Recall_Port_sweep    = TP_Port_sweep/(TP_Port_sweep+FN_Port_sweep)
    if((((Beta**2)*Precision_Normal)+Recall_Normal)==0):
        Fbeta_Normal = 0 
    else:   
        Fbeta_Normal = (1+(Beta**2))*Precision_Normal*Recall_Normal/(((Beta**2)*Precision_Normal)+Recall_Normal)
    if((((Beta**2)*Precision_DOS)+Recall_DOS)==0):
        Fbeta_DOS = 0
    else:
        Fbeta_DOS = (1+(Beta**2))*Precision_DOS*Recall_DOS/(((Beta**2)*Precision_DOS)+Recall_DOS)
    if((((Beta**2)*Precision_IP_sweep)+Recall_IP_sweep)==0):
        Fbeta_IP_sweep = 0
    else:
        Fbeta_IP_sweep = (1+(Beta**2))*Precision_IP_sweep*Recall_IP_sweep/(((Beta**2)*Precision_IP_sweep)+Recall_IP_sweep)
    if((((Beta**2)*Precision_Nmap)+Recall_Nmap)==0):
        Fbeta_Nmap = 0
    else:
        Fbeta_Nmap = (1+(Beta**2))*Precision_Nmap*Recall_Nmap/(((Beta**2)*Precision_Nmap)+Recall_Nmap)
    if((((Beta**2)*Precision_Port_sweep)+Recall_Port_sweep)==0):
        Fbeta_Port_sweep = 0
    else:
        Fbeta_Port_sweep = (1+(Beta**2))*Precision_Port_sweep*Recall_Port_sweep/(((Beta**2)*Precision_Port_sweep)+Recall_Port_sweep)
    Macro_Fbeta = (Fbeta_Normal + Fbeta_DOS + Fbeta_IP_sweep + Fbeta_Nmap + Fbeta_Port_sweep)/5
    if(total_cost==0):
        Evaluate_criteria = Alpha + (1-Alpha) * Macro_Fbeta
    else:
        Evaluate_criteria = Alpha * (1-(math.log(total_cost)/math.log(max_cost))) + (1-Alpha) * Macro_Fbeta
    #print("TN_Normal =     ",TN_Normal,     ",TP_Normal =     ",TP_Normal ,    ",FN_Normal =     ",FN_Normal,     ",FP_Normal =     ",FP_Normal)
    #print("TN_DOS =        ",TN_DOS,        ",TP_DOS =        ",TP_DOS ,       ",FN_DOS =        ",FN_DOS,        ",FP_DOS =        ",FP_DOS)
    #print("TN_IP_sweep =   ",TN_IP_sweep,   ",TP_IP_sweep =   ",TP_IP_sweep ,  ",FN_IP_sweep =   ",FN_IP_sweep,   ",FP_IP_sweep =   ",FP_IP_sweep)
    #print("TN_Nmap =       ",TN_Nmap,       ",TP_Nmap =       ",TP_Nmap ,      ",FN_Nmap =       ",FN_Nmap,       ",FP_Nmap =       ",FP_Nmap)
    #print("TN_Port_sweep = ",TN_Port_sweep, ",TP_Port_sweep = ",TP_Port_sweep ,",FN_Port_sweep = ",FN_Port_sweep, ",FP_Port_sweep = ",FP_Port_sweep)
    #print("total_cost = ",total_cost,",max_cost = ",max_cost)
    return Evaluate_criteria

#r = random.randint(3,100)

#a = np.random.randint(5,size=r)
#b = np.random.randint(5,size=r)


#print("predicted: ",a)
#print("real:      ",b)
#print(score(a,b))
