from numpy import *
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import spsolve

def readMsh4_1(fn):
    '''
    Read data from a GMSH 4.1 MSH file
    Outputs:
        nodes: x, y, z
        elem:  n1, n2, n3, dom_ix
        bc:    n1, n2, bc_ix
        lab:   {1:'name1',2:'name2', ...}
    '''
    msh = open(fn)
    
    pn = dict()
    while not "$PhysicalNames" in msh.readline().rstrip():
        pass
    for i in range(int(msh.readline().rstrip())):
        dim, tag, name = msh.readline().rstrip().split()
        if int(dim) == 1:
            pn[int(tag)] = name.strip('"')
        elif int(dim) == 2:
            pn[int(tag)] = name.strip('"')

    cix = dict()
    six = dict()
    while not "$Entities" in msh.readline().rstrip():
        pass
    np, nc, ns, nv = map(int,msh.readline().rstrip().split())
    for i in range(np):
        msh.readline()
    for i in range(nc):
        info = msh.readline().rstrip().split()
        cix[int(info[0])] = int(info[int(info[7])+7])
    for i in range(ns):
        info = msh.readline().rstrip().split()
        six[int(info[0])] = int(info[int(info[7])+7])
        
    nodes = list()
    while not "$Nodes" in msh.readline().rstrip():
        pass
    ne, nn, nmin, nmax = map(int,msh.readline().rstrip().split())
    for i in range(ne):
        ed, et, p, numn = map(int,msh.readline().rstrip().split())
        for j in range(numn):
            msh.readline()
        for j in range(numn):
            x,y,z = map(float,msh.readline().rstrip().split())
            nodes.append((x,y,z))
    nodes = array(nodes)

    elem = list()
    bced = list()
    while not "$Elements" in msh.readline().rstrip():
        pass
    ne, nelm, emin, emax = map(int,msh.readline().rstrip().split())
    for i in range(ne):
        ed, et, etype, nume = map(int,msh.readline().rstrip().split())
        if etype == 1:
            for j in range(nume):
                num,a,b = map(int,msh.readline().rstrip().split())
                bced.append((a-1,b-1,cix[et]))
        elif etype == 2:
            for j in range(nume):
                num,a,b,c = map(int,msh.readline().rstrip().split())
                if cross(nodes[b-1]-nodes[a-1],nodes[c-1]-nodes[a-1])[2] > 0:
                    elem.append((a-1,b-1,c-1,six[et]))
                else:
                    elem.append((a-1,c-1,b-1,six[et]))
        else:
            for j in range(nume):
                msh.readline()
    bced = array(bced)
    elem = array(elem)

    return nodes,elem,bced,pn



def axiHeatCond(mshfile, domain, boundary):
    '''
    Input:
        mshfile: Name of GMSH output "msh" file
        domain: List of domain names and conductivity coefficients:
            {'name1':k1, 'name2':k2, ...}
        boundary: List of boundary names with alpha and beta defined
            {'name1':(a1,b1), 'name2':(a2,b2), ...}

    Output:
        x: Node coordinate x values
        y: Node coordinate y values
        tri: List of triangle indices
        T: Computed temperature
        V: Volumes of the domains in dictionary form
        q: Area and heat flow for the boundaries, in dictionary form

    Note that the y-axis represents the radius from centre
    '''
    nodes, elem, bc, lab = readMsh4_1(mshfile)

    A = dok_matrix((len(nodes),len(nodes)))
    volume = dict()
    for e in elem:
        dlab = lab[e[3]]
        a = nodes[roll(e[:3],1)]-nodes[roll(e[:3],-1)]
        b = cross(a,[0,0,1])
        S = cross(a[2],-a[1])[2] / 2.0
        if S < 0:
            print("ERROR")
        r = mean(nodes[e[:3]][:,1])
        ix = outer(e[:3],[1,1,1])
        k = domain[dlab]
        A[ix,ix.T] += k * (b @ b.T) / (4*S) * 2*pi*r
        volume[dlab] = volume.get(dlab, 0.0) + 2*pi*r*S

    rhs = zeros(len(nodes))
    for b in bc:
        a = linalg.norm(nodes[b[0]]-nodes[b[1]])
        r = mean(nodes[b[:2]][:,1])
        ix = outer(b[:2],[1,1])
        alpha, beta = boundary[lab[b[2]]]
        A[ix,ix.T] += alpha * a * array([[1,2],[2,1]]) / 6 * 2*pi*r
        rhs[b[:2]] -= beta * a * array([1,1]) / 2 * 2*pi*r
        
    A = A.tocsr()
    T = spsolve(A,rhs)

    edges = dict()
    for b in bc:
        blab = lab[b[2]]
        r = mean(nodes[b[:2]][:,1])
        area = 2*pi*r*linalg.norm(nodes[b[0]]-nodes[b[1]])
        alpha, beta = boundary[blab]
        q = alpha * mean(T[b[:2]]) + beta
        if blab in edges:
            edges[blab][0] += area
            edges[blab][1] += area*q
        else:
            edges[blab] = [area,area*q]
    
    return nodes[:,0], nodes[:,1], elem[:,:3], T, volume, edges

