using ITensors
using ITensors: Algorithm, @Algorithm_str
using LinearAlgebra

function makeW(::Algorithm"WI", ElT, t, A, B, C, D)
    # See: https://github.com/tenpy/tenpy/blob/main/tenpy/networks/mpo.py#L1499
    tC = sqrt(abs(t))
    tB = t / tC
    
    s = commonind(C, D)
    ri = noncommonind(A, C)
    ci = noncommonind(A, B)
    
    d = dim(s)
    #The virtual size of W is  (1+Nr, 1+Nc)
    Nr = dim(ri)
    Nc = dim(ci)
    W = zeros(ElT, 1 + Nr, 1 + Nc, d, d)

    W[1,1,:,:] = I(d) + t * Array(D, s, s')

    for r in eachval(ri), c in eachval(ci)
        W[1+r, 1+c, :, :] = Array(A * onehot(dag(ci) => c, dag(ri) => r), s, s')
    end
    for c in eachval(ci)
        W[1, 1+c, :, :] = Array(tC * (C * onehot(dag(ci) => c)), s, s')
    end
    for r in eachval(ri)
        W[1+r, 1, :, :] = Array(tB * (B * onehot(dag(ri) => r)), s, s')
    end
    
    W
end

function makeW(::Algorithm"WII", ElT, t, A, B, C, D)
    # See: https://github.com/tenpy/tenpy/blob/main/tenpy/networks/mpo.py#L1499
    tC = sqrt(abs(t))  #spread time step across B, C
    tB = t / tC
    
    s = commonind(C, D)
    ri = noncommonind(A, C)
    ci = noncommonind(A, B)
    
    d = dim(s)
    #The virtual size of W is  (1+Nr, 1+Nc)
    Nr = dim(ri)
    Nc = dim(ci)

    W = zeros(ElT, 1 + Nr, 1 + Nc, d, d)

    # construct indices for the two auxiliary bosons 
    i1 = Index(2, "first boson") # a
    i2 = Index(2, "second boson") # abar

    cd1 = onehot(ElT, i1 => 1, i1' => 2) # cdag_a
    cd2 = onehot(ElT, i2 => 1, i2' => 2) # cbardag_abar

    ket00 = onehot(ElT, i1 => 1, i2 => 1)

    bra00 = onehot(ElT, i1' => 1, i2' => 1)
    bra01 = onehot(ElT, i1' => 1, i2' => 2)
    bra10 = onehot(ElT, i1' => 2, i2' => 1)
    bra11 = onehot(ElT, i1' => 2, i2' => 2)

    Id = delta(ElT, i1', i1) * delta(ElT, i2', i2)
    Br = cd1 * delta(ElT, i2', i2)
    Bc = delta(ElT, i1', i1) * cd2
    Brc = cd1 * cd2

    for r in eachval(ri)  #double loop over row / column of A
        for c in eachval(ci)
            h = Brc * (A * onehot(ElT, dag(ci) => c, dag(ri) => r)) + Br * tB * (B * onehot(ElT, dag(ri) => r)) + Bc * tC * (C * onehot(ElT, dag(ci) => c)) + t * Id * D
            w = exp(h) * ket00
            W[1+r, 1+c, :, :] = Array(bra11 * w, s, s')
            if c == 1
                W[1+r, 1, :, :] = Array(bra10 * w, s, s')
            end
            if r == 1
                W[1, 1+c, :, :] = Array(bra01 * w, s, s')
                if c == 1
                    W[1, 1, :, :] = Array(bra00 * w, s, s')
                end
            end
        end
        if Nc == 0  #technically only need one boson
            h = Br * tB * (B * onehot(ElT, dag(ri) => r)) + t * Id * D
            w = exp(h) * ket00
            W[1+r, 1, :, :] = Array(bra10 * w, s, s')
            if r == 1
                W[1, 1, :, :] = Array(bra00 * w, s, s')
            end
        end
    end
    if Nr == 0
        for c in eachval(ci)
            h = Bc * tC * (C * onehot(ElT, dag(ci) => c)) + t * Id * D
            w = exp(h) * ket00
            W[1, 1+c, :, :] = Array(bra01 * w, s, s')
            if c == 1
                W[1, 1, :, :] = Array(bra00 * w, s, s')
            end
        end
        if Nc == 0
            W = reshape(Array(exp(t * D), s, s'), 1, 1, d, d)
        end
    end

    W
end