from __main__ import *


def M_step(pdf):
    w=w(pdf)
    m=u(pdf)
    c=sig(m,pdf)
    return w,m,c

def w(pdf):
    psum=[]
    for i in range(K):
        psum.append(sum(pdf[i,:]))
    w=psum/sum(psum)
    return w

def u(pdf):
    psum=[]
    m=np.matmul(pdf.T,x)
    for i in range(K):
        psum.append(sum(pdf[i,:]))
        m[i]=m[i]/psum[i]
    return m

def sig(m, pdf):
    psum = []
    kvars=[]
    for i in range(K):
        psum.append(sum(pdf[i, :]))
        a=[]
        for j in range (cols-1):
            c=[]
            for v in range(cols - 1):
                verify=0
                for ii in range(row):
                    F=x[ii,j]-m[i,j]
                    S=x[ii,v]-m[i,v]
                    verify+=pdf[ii,i]*F*S
                verify=verify/psum[i]
                if j==v:
                    if abs(verify)<=0.0001:
                        verify=0.0001
                c.append(verify)
            a.append(c)
        kvars.append[a]
    return kvars





