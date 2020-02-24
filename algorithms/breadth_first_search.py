BFS(v):
    Q = Queue()
    Q.add(v)
    visited = set()
    visted.add(v) #Include v in the set of elements already visited
    while (not IsEmpty(Q)):
        w = Q.dequeue() #Remove a single element from Q and store it in w
        for u in w.vertices(): #Go through every node w is adjacent to.
            if (not u in visited): #Only do something if u hasn't already been visited
                visited.add(u)
                Q.add(u)