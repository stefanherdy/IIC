#!/usr/bin/python
import torch

def transform(input, randlist_all):

    flipprob = 0.5
    flipproblr = 0.5
    rot = 0.5

    partstup = torch.split(input, 1)
    parts = []

    for i in range(len(partstup)):
        parts.append(partstup[i])

    
    for i in range(len(parts)):
        randlist = randlist_all[i] 
        if randlist[0] < 10*flipprob:
            parts[i] = torch.flip(parts[i], [2])
        else:
            parts[i] = parts[i]
            
        if randlist[1] < 10*flipproblr:
            parts[i] = torch.flip(parts[i], [3])
        else:
            parts[i] = parts[i]
        
        if randlist[2] < 10*rot:
            parts[i] = torch.rot90(parts[i], 1, [2, 3])
        else:
            parts[i] = parts[i]

        if randlist[3] < 10*rot:
            parts[i] = torch.rot90(parts[i], 3, [2, 3])
        else:
            parts[i] = parts[i]

        if randlist[4] < 10*rot:
            parts[i] = torch.rot90(parts[i], 2, [2, 3])
        else:
            parts[i] = parts[i]
        
    target = torch.cat(parts)
            
    return target


def backtransform(input, randlist_all):

    flipprob = 0.5
    flipproblr = 0.5
    rot = 0.8

    partstup = torch.split(input, 1)
    parts = []

    for i in range(len(partstup)):
        parts.append(partstup[i])

    
    for i in range(len(parts)):
        randlist = randlist_all[i] 
        if randlist[0] < 10*flipprob:
            parts[i] = torch.flip(parts[i], [2])
        else:
            parts[i] = parts[i]
            
        if randlist[1] < 10*flipproblr:
            parts[i] = torch.flip(parts[i], [3])
        else:
            parts[i] = parts[i]

        if randlist[2] < 10*rot:
            parts[i] = torch.rot90(parts[i], 3, [2, 3])
        else:
            parts[i] = parts[i]
        
        if randlist[3] < 10*rot:
            parts[i] = torch.rot90(parts[i], 1, [2, 3])
        else:
            parts[i] = parts[i]

        if randlist[4] < 10*rot:
            parts[i] = torch.rot90(parts[i], 2, [2, 3])
        else:
            parts[i] = parts[i]

    target = torch.cat(parts)

    return target

def get_colours():
    red = [255,0,0]
    green = [0,255,0]
    blue = [0,0,255]
    yellow = [255,255,0]
    black = [0,0,0]
    white = [255,255,255]
    cyan = [0,255,255]
    orange = [255,128,0]
    pink = [255,0,255]
    violett = [102,0,204]
    dark_green = [0,102,0]

    colours = [red, green, blue, yellow, black, white, cyan, orange, pink, violett, dark_green]
    return colours

    