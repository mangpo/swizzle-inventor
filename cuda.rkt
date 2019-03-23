#|
 | Copyright (c) 2018-2019, University of California, Berkeley.
 | Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
 |
 | Redistribution and use in source and binary forms, with or without 
 | modification, are permitted provided that the following conditions are met:
 |
 | 1. Redistributions of source code must retain the above copyright notice, 
 | this list of conditions and the following disclaimer.
 |
 | 2. Redistributions in binary form must reproduce the above copyright notice, 
 | this list of conditions and the following disclaimer in the documentation 
 | and/or other materials provided with the distribution.
 |
 | THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 | AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 | IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 | ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
 | LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 | CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 | SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 | INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 | CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 | ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 | POSSIBILITY OF SUCH DAMAGE.
 |#

#lang rosette

(require "util.rkt")

(define (drop@ name) 
  (if (regexp-match? #rx"^@.+$" name)
      (regexp-replace #rx"@" name "")
      name))

(require (only-in racket [sort %sort] [< %<]))
(provide (rename-out [@+ +] [@- -] [@* *] [@modulo modulo] [@quotient quotient] [@< <] [@<= <=] [@> >] [@>= >=] [@= =] [@ite ite]
                     [@bvadd bvadd] [@bvsub bvsub] [@bvand bvand] [@bvxor bvxor] [@bvshl bvshl] [@bvlshr bvlshr] [@extract extract] [@bvlog bvlog])
         @int @bv @dup gen-uid gen-sym gen-bv for/bounded
         define-shared
         create-matrix-local
         global-to-shared shared-to-global
         global-to-local local-to-global
         global-to-reg reg-to-global reg-to-global-update
         warpSize set-warpSize blockSize set-blockSize
         get-warpId get-idInWarp get-blockDim get-gridDim get-global-threadId
         shfl shfl-send sw-xform sw-xform-prime rotate-nogroup permute-vector
         accumulator accumulator? accumulator-val create-accumulator accumulate accumulate-merge accumulate-final
         get-accumulator-val acc-equal? acc-print
         run-kernel get-cost reset-cost)


(define warpSize 4)
(define blockSize warpSize)
(define blockDim (list blockSize))
(define gridDim (list 1))

;; Return a vector of size blockSize with value x.
(define (@dup x) (for/vector ([i blockSize]) x))
(define (get-blockDim) blockDim)
(define (get-gridDim) gridDim)

(define (set-warpSize s)
  (set! warpSize s))
(define (set-blockSize s)
  (set! blockSize s))

(define uid 0)

;; Generate a unique id.
(define (gen-uid)
  (set! uid (add1 uid))
  uid)

;; Generate a symbolic integer variable.
(define (gen-sym)
  (define-symbolic* x integer?)
  x)

;; Generate a symbolic bitvector variable.
(define (gen-bv)
  (define-symbolic* x (bitvector 4))
  x)

;;;;;;;;;;;;;;;;;;;;;;;;;;; lifted operations ;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Create a variable in shared memory.
(define-syntax-rule (define-shared x exp) (define x exp))

;; Apply op on every element of x and y.
(define (iterate x y op)
  (define (f x y)
    (cond
      [(and (vector? x) (vector? y)) (for/vector ([i (vector-length x)]) (f (get x i) (get y i)))]
      [(vector? x) (for/vector ([i (vector-length x)]) (f (get x i) y))]
      [(vector? y) (for/vector ([i (vector-length y)]) (f x (get y i)))]
      [(and (list? x) (list? y)) (map f x y)]
      [(list? x) (map (lambda (xi) (f xi y)) x)]
      [(list? y) (map (lambda (yi) (f x yi)) y)]
      [else (op x y)]))
  (f x y))

;; Vectorized ite
(define (@ite c x y) ;; TODO: not quite correct
  (define (f c x y)
    (cond
      [(and (vector? c) (vector? x) (vector? y)) (for/vector ([i (vector-length c)]) (f (get c i) (get x i) (get y i)))]
      [(and (vector? c) (vector? x)) (for/vector ([i (vector-length c)]) (f (get c i) (get x i) y))]
      [(and (vector? c) (vector? y)) (for/vector ([i (vector-length c)]) (f (get c i) x (get y i)))]
      [(and (vector? x) (vector? y)) (for/vector ([i (vector-length x)]) (f c (get x i) (get y i)))]
      [(and (vector? c)) (for/vector ([i (vector-length c)]) (f (get c i) x y))]
      [(and (vector? x)) (for/vector ([i (vector-length x)]) (f c (get x i) y))]
      [(and (vector? y)) (for/vector ([i (vector-length y)]) (f c x (get y i)))]
      [else (if c x y)])
    )
  (f c x y))

(define-syntax-rule (define-operator my-op @op op)
  (begin
    (define (@op l)
      (define ret
        (cond
          [(= (length l) 1) (car l)]
          [(= (length l) 2)
           (iterate (first l) (second l) op)]
          [else (iterate (first l) (@op (cdr l)) op)]))
      (inc-cost my-op ret l)
      ret)
    (define my-op (lambda l (@op l)))
    ))

;; Vector operations with cost 0.
(define-operator @++ $++ +)
(define-operator @** $** *)

;; Vector operations.
(define-operator @+ $+ +)
(define-operator @- $- -)
(define-operator @* $* *)
(define-operator @> $> >)
(define-operator @>= $>= >=)
(define-operator @< $< <)
(define-operator @<= $<= <=)
(define-operator @= $= =)
(define-operator @modulo $modulo modulo)
(define-operator @quotient $quotient quotient)

(define-operator @bvadd $bvadd bvadd)
(define-operator @bvsub $bvsub bvsub)
(define-operator @bvand $bvand bvand)
(define-operator @bvxor $bvxor bvxor)
(define-operator @bvshl $bvshl bvshl)
(define-operator @bvlshr $bvlshr bvlshr)

(define (@bv x)
  (if (vector? x)
      (for/vector ([i (vector-length x)])
        (@bv (vector-ref x i)))
      (integer->bitvector x (bitvector BW))))

(define (@int x)
  (if (vector? x)
      (for/vector ([i (vector-length x)])
        (@int (vector-ref x i)))
      (bitvector->integer x)))

(define (@bvlog x)
  (define y (log x 2))
  (assert (integer? y))
  (integer->bitvector (exact->inexact y) (bitvector BW)))

(define (@extract x b)
  (if (vector? x)
      (for/vector ([i (vector-length x)])
        (@extract (vector-ref x i) b))
      (let ([s (bvsub (bv BW (bitvector BW)) b)])
        (bvlshr (bvshl x s) s))))

;; Compute GCD of x and y with recursive bound = 8.
(define (gcd/bound x y [depth 8])
  (assert (> depth 0))
  (if (= y 0)
      x
      (gcd/bound y (modulo x y) (sub1 depth))))

;; Produce a permutation of 1D vector x of size n according to
;; the shuffle function f.
(define (permute-vector x n f)
  (pretty-display `(permute-vector ,n))
  (define y (create-matrix-local (x-y-z n)))
  (for ([i n])
    (pretty-display `(i ,i ,n))
    (set y (@dup i) (get x (f i))))
  y)

;; Transformation index swizzle.
;; Refer to Section 5.3 of https://mangpo.net/papers/swizzle-inventor-asplos19.pdf
;; The arguments' names should be consistent with the paper.
(define (sw-xform i n cf df group wrap 
             k m cr dr [c 0]
             #:gcd [gcd (quotient group df)]
             #:ecr [ecr 0] #:ec [ec 0] ; extra rot
             ;;#:cz [cz 1] #:nz [nz group] ; extra fan
             )
  (assert (and (>= group 1) (<= group n)))
  (assert (and (>= cf -1) (< cf group)))
  (assert (and (>= cr -1) (< cr group)))
  (assert (and (>= c 0) (< c group)))
  
  (define rem (modulo n group))
  (assert (= rem 0))

  ;; df should be group/gcd(group, cf)
  ;; gcd = group/df
  (assert (= (modulo group df) 0))
  (assert (= (modulo cf (quotient group df)) 0))
  
  (assert (= (modulo m dr) 0))
  ;; If we don't impose gcd to be actual gcd(group, cf), then our equation contains Eq(24) from Trove.
  ;; (assert (= gcd (quotient group df)))

  (define ii (@modulo (@+ i (@* ecr k) ec) group)) ; extra rot (before fan)
  ;; (define ii (@modulo (@+ (@* j cz) (@quotient j nz)) group)) ; extra fan

  (define offset1 (@+ (@quotient ii df) ; fan conflict
                      (@* k cr) (@quotient k dr) c))    ; rot
  (define offset2 ; rotation (after fan)
    (if (= wrap 1)
        offset1   ; rot
        (@modulo offset1 gcd))) ; grouped rot
  
  (@+ (@* (@quotient ii group) group) ; top-level group 
      (@modulo (@+ (@* ii cf) ; fan without fan conflict
                   offset2)   ; fan conflict + rotation
               group))
  )

;; sw-xform when cf and n are co-prime.
(define-syntax-rule (sw-xform-prime i n cf
                               k m cr dr)
  (sw-xform i n cf n n 1
       k m cr dr))

;; rotation
(define-syntax-rule (rotate-nogroup i n 
                                    k m cr dr)
  (sw-xform i n 1 n n 1
       k m cr dr))

;;;;;;;;;;;;;;;;;;;;;;;;;;; performance cost ;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define cost 0)
(define (reset-cost) (set! cost 0))
(define (get-cost) cost)

(define (cost-of op)
  (cond
    [(member op (list @+ @- @> @>= @< @<= @= @bvadd @bvsub @bvand @bvxor @bvshl @bvlshr)) 1]
    [(member op (list @* @modulo @quotient)) 2]
    [(member op (list @++ @**)) 0]
    [else (assert `(cost-of ,op unimplemented))]))

(define (zero? x) (= x 0))
(define (one? x) (= x 1))
(define (minus-one? x) (= x -1))
(define (zero-bv? x) (= x (bv 0 BW)))
(define (one-bv? x) (= x (bv 1 BW)))
(define (minus-one-bv? x) (= x (bv -1 BW)))
(define (true? x) (and (boolean? x) x))
(define (false? x) (and (boolean? x) (not x)))

(define (all? x f)
  (cond
    [(vector? x)
     (define ret #t)
     (for ([i (vector-length x)])
       (set! ret (and ret (all? (vector-ref x i) f))))
     ret]

    [(list? x)
     (andmap (lambda (xi) (all? xi f)) x)]

    [else (f x)]))

(define (size-of x)
   (cond
    [(vector? x)
     (define len (vector-length x))
     (if (> len 0)
         (* len (size-of (vector-ref x 0)))
         0)]

    [(list? x)
     (define len (length x))
     (if (> len 0)
         (* len (size-of (car x)))
         0)]

    [else 1]))

(define (inc-cost op ret args)
  (define op-cost (cost-of op))
  
  (define inc
    (cond
      [(member op (list @+ @-))
       (cond
         [(all? (first args) zero?) 0]
         [(all? (second args) zero?) 0]
         [(all? ret zero?) 0]
         [else op-cost])]

      [(member op (list @modulo))
       (cond
         [(all? (second args) one?) 0]
         [else op-cost])]
      
      [(member op (list @*))
       (cond
         [(all? (first args) zero?) 0]
         [(all? (first args) one?) 0]
         [(all? (first args) minus-one?) 0]
         [(all? (second args) zero?) 0]
         [(all? (second args) one?) 0]
         [(all? (second args) minus-one?) 0]
         [else op-cost])]
      
      [(member op (list @quotient))
       (cond
         [(all? (second args) one?) 0]
         [else op-cost])]
      
      [(member op (list @bvadd @bvsub))
       (cond
         [(all? ret zero-bv?) 0]
         [else op-cost])]
      
      [(member op (list @bvshl @bvlshr))
       (cond
         [(all? (second args) zero-bv?) 0]
         [else op-cost])]

      [else op-cost]
      ))
  ;;(set! cost (+ cost inc))
  (void)
  )

(define (accumulate-cost ops vals)
  (define (f ops vals)
    (cond
      [(vector? vals)
       (* (vector-length vals)
          (+ (cost-of (car ops)) (f (cdr ops) (vector-ref vals 0))))]
      
      [(list? vals)
       (* (length vals)
          (+ (cost-of (car ops)) (f (cdr ops) (car vals))))]
      
      [(empty? ops) 0]
      [else (cost-of (car ops))]
      ))

  (define inc
    (cond
      [(vector? vals)
       (+ (cost-of (last ops))
          (f (cdr (reverse ops)) (vector-ref vals 0)))]
      
      [else
       (f (reverse ops) vals)]))
  ;;(set! cost (+ cost inc))
  (set! cost (+ cost 1))
  )

(define (global-cost pattern sizes)
  (define pattern-x (get-x pattern))
  (define my-cost
    (if (= pattern-x 1)
        (+ 1 (quotient (apply * sizes) blockSize))
        (* 4 (+ 1 (quotient (apply * sizes) blockSize)))))
  ;;(set! cost (+ cost my-cost))
  (void)
  )

;;;;;;;;;;;;;;;;;;;;;;;;;;; memory operations ;;;;;;;;;;;;;;;;;;;;;;;;;;;
(define-syntax-rule
  (for/bounded ([i I]) body ...)
  (letrec ([f (lambda (i bound)
                (when (< i I)
                  (if (> bound 0)
                      (begin
                        body ...
                        (f (+ i 1) (- bound 1)))
                      (assert #f))))])
    (f 0 8)))

;; Create a local matrix.
(define (create-matrix-local dims [init (lambda () 0)])
  (create-matrix (append dims (list blockSize))))

;; Load I in global memory to I-shared in shared memory
;; pattern -- (x-y-z stride-x ...)
;;   >> each thread load stride-x * stride-y * ... consecutive block in round-robin fasion
;; offset -- the starting x-y-z coordinate of global memory that the thread block loads.
;; sizes  -- (x-y-z size-x ...)
;;   >> each thread block loads size-x * size-y * ... values
;; transpose -- #t for load with transpose
;; round -- (x-y-z round-x ...) or just round-x for 1D. Round of the round robin to fully load 'sizes'.
;; gsize -- (x-y-z gsize-x ...) size of global memory, must be specified for 2D and 3D
(define (global-to-shared I I-shared pattern offset sizes [transpose #f]
                          #:round [round 1] #:size [gsize #f])
  (global-cost pattern sizes)
  (define bounds (get-dims I))
  (pretty-display `(sizes ,sizes))
  (pretty-display `(bounds ,(@* blockDim pattern round)))
  (assert (all? (@<= sizes (@* blockDim pattern round)) true?) "size 1")
  (assert (all? (@> sizes (@* blockDim pattern (@- round 1))) true?) "size 2")
  (when (> (length pattern) 1) (assert gsize "#:size must be specified for dimenion > 1"))
  
  (cond
    [(= (length offset) 1)
     (let ([size-x (get-x sizes)]
           [bound-x (get-x bounds)]
           [offset-x (get-x offset)])
       (when (vector? offset-x) (set! offset-x (vector-ref offset-x 0)))
       (for ([i size-x])
         (when (< (+ offset-x i) bound-x)
           (set I-shared i (get I (+ offset-x i))))))]
    
    [(= (length offset) 2)
     (let ([size-x (get-x sizes)]
           [size-y (get-y sizes)]
           [bound-x (get-x bounds)]
           [bound-y (get-y bounds)]
           [offset-x (get-x offset)]
           [offset-y (get-y offset)])
       (when (vector? offset-x)
         (set! offset-x (vector-ref offset-x 0))
         (set! offset-y (vector-ref offset-y 0))
         )
       (for* ([y size-y] [x size-x])
         (when (and (< (+ offset-x x) bound-x) (< (+ offset-y y) bound-y))
           (if transpose
               (set I-shared y x (get I (+ offset-x x) (+ offset-y y)))
               (set I-shared x y (get I (+ offset-x x) (+ offset-y y)))))))]
    
    [(= (length offset) 3)
     (let ([size-x (get-x sizes)]
           [size-y (get-y sizes)]
           [size-z (get-z sizes)]
           [bound-x (get-x bounds)]
           [bound-y (get-y bounds)]
           [bound-z (get-z bounds)]
           [offset-x (get-x offset)]
           [offset-y (get-y offset)]
           [offset-z (get-z offset)])
       (when (vector? offset-x)
         (set! offset-x (vector-ref offset-x 0))
         (set! offset-y (vector-ref offset-y 0))
         (set! offset-z (vector-ref offset-z 0))
         )
       (for* ([z size-z] [y size-y] [x size-x])
         (when (and (< (+ offset-x x) bound-x) (< (+ offset-y y) bound-y) (< (+ offset-z z) bound-z))
           (if transpose
               (set I-shared z y x (get I (+ offset-x x) (+ offset-y y) (+ offset-z z)))
               (set I-shared x y z (get I (+ offset-x x) (+ offset-y y) (+ offset-z z)))))))]
    ))

;; Similar to global-to-shared but
;; for storing I-shared in shared memory to I in global memory
(define (shared-to-global I-shared I pattern offset sizes [transpose #f] #:round [round 1] #:size [s #f])
  (if transpose
      (global-cost (reverse pattern) (reverse sizes))
      (global-cost pattern sizes))
  (define bounds (get-dims I))
  (assert (all? (@<= sizes (@* blockDim pattern round)) true?))
  (assert (all? (@> sizes (@* blockDim pattern (@- round 1))) true?))
  
  (cond
    [(= (length offset) 1)
     (let ([size-x (get-x sizes)]
           [bound-x (get-x bounds)]
           [offset-x (get-x offset)])
       (for ([i size-x])
         (when (< (+ offset-x i) bound-x)
           (set I (+ offset-x i) (get I-shared i)))))]
    
    [(= (length offset) 2)
     (let ([size-x (get-x sizes)]
           [size-y (get-y sizes)]
           [bound-x (get-x bounds)]
           [bound-y (get-y bounds)]
           [offset-x (get-x offset)]
           [offset-y (get-y offset)])
       (for* ([y size-y] [x size-x])
         (when (and (< (+ offset-x x) bound-x) (< (+ offset-y y) bound-y))
           (if transpose
               (set I (+ offset-x y) (+ offset-y x) (get I-shared x y))
               (set I (+ offset-x x) (+ offset-y y) (get I-shared x y))))))]
    
    [(= (length offset) 3)
     (let ([size-x (get-x sizes)]
           [size-y (get-y sizes)]
           [size-z (get-z sizes)]
           [bound-x (get-x bounds)]
           [bound-y (get-y bounds)]
           [bound-z (get-z bounds)]
           [offset-x (get-x offset)]
           [offset-y (get-y offset)]
           [offset-z (get-z offset)])
       (for* ([z size-z] [y size-y] [x size-x])
         (when (and (< (+ offset-x x) bound-x) (< (+ offset-y y) bound-y) (< (+ offset-z z) bound-z))
           (if transpose
               (set I (+ offset-x z) (+ offset-y y) (+ offset-z x) (get I-shared x y z))
               (set I (+ offset-x x) (+ offset-y y) (+ offset-z z) (get I-shared x y z))))))]
    ))

;; Load I in global memory at offset to register I-reg.
;; gsize -- (x-y-z gsize-x ...) size of global memory, must be specified for 2D and 3D
(define-syntax global-to-reg
    (syntax-rules ()
      ((global-to-reg I I-reg offset)
       (let* ([bounds (get-dims I)]
              [blockSize (vector-length offset)]
              [new-I-reg (make-vector blockSize #f)])
         (global-cost (list 1) (list (size-of I-reg)))
         (for ([t blockSize])
           (set new-I-reg t (clone I-reg)))
         (set! I-reg new-I-reg)
         (for ([i blockSize]
               [global-i offset])
           (when (for/and ([b bounds] [i global-i]) (< i b))
             (set I-reg i (get* I global-i))))))

      ((global-to-reg I I-reg offset #:size gsize)
       (global-to-reg I I-reg offset))))


;; Store register I-reg to I in global memory at offset.
;; gsize -- (x-y-z gsize-x ...) size of global memory, must be specified for 2D and 3D
(define (reg-to-global I-reg I offset #:size [gsize #f])
  (let* ([bounds (get-dims I)]
         [blockSize (vector-length offset)])
    (global-cost (list 1) (list (size-of I-reg)))
    (for ([i blockSize]
          [global-i offset])
      (when (for/and ([b bounds] [i global-i]) (< i b))
        (set* I global-i (get I-reg i))))))

;; Update I in global memory at offset to f(old_value, I-reg)
;; gsize -- (x-y-z gsize-x ...) size of global memory, must be specified for 2D and 3D
(define (reg-to-global-update f I-reg I offset #:size [gsize #f] #:pred [pred (make-vector blockSize)])
  (let* ([bounds (get-dims I)]
         [blockSize (vector-length offset)])
    (global-cost (list 1) (list (size-of I-reg)))
    (for ([i blockSize]
          [global-i offset])
      (when (and (vector-ref pred i)
                 (for/and ([b bounds] [i global-i])
                   (< i b)))
        (set* I global-i (f (get* I global-i) (get I-reg i)))))))

;; Load I in global memory to I-reg in local memory/registers
;; pattern -- (x-y-z stride-x ...)
;;   >> each thread load stride-x * stride-y * ... consecutive block in round-robin fasion
;; offset -- the starting x-y-z coordinate of global memory that the warp loads.
;; sizes  -- (x-y-z size-x ...)
;;   >> each warp loads size-x * size-y * ... values
;; transpose -- #t for load with transpose
;; warp-shape -- (x-y-z shape-x shape-y ...) must be specified for 2D and 3D
;; round -- (x-y-z round-x ...) or just round-x for 1D. Round of the round robin to fully load 'sizes'.
;; shfl  -- shuffle function for load with shuffle. 'k' is the iteration of the round robin.
;; gsize -- (x-y-z gsize-x ...) size of global memory, must be specified for 2D and 3D
(define (global-to-local I I-reg pattern offset sizes transpose
                         #:warp-shape [warp-shape warpSize]
                         #:round [round 1]
                         #:shfl [shfl (lambda (tid k) tid)]
                         #:size [gsize #f])
  (global-cost pattern sizes)
  (assert (all? (@<= sizes (@* warp-shape pattern round)) true?))
  (assert (all? (@> sizes (@* warp-shape pattern (@- round 1))) true?))
  (cond
    [(= (length blockDim) 1)
     (let* ([size-x (get-x sizes)]
            [stride-x (get-x pattern)]
            [blockSize (apply * blockDim)]
            [iter-x (add1 (quotient (sub1 size-x) (* warpSize stride-x)))]
            [I-len (vector-length I)]
            [I-reg-len (vector-length (vector-ref I-reg 0))])
       (for ([warp (quotient blockSize warpSize)])
         (let ([offset-x (if (vector? offset)
                             (get-x (vector-ref offset (* warp warpSize)))
                             (vector-ref (get-x offset) (* warp warpSize)))])
           ;(pretty-display `(offset-x ,offset-x))
           (for/bounded ([it iter-x])
             (for ([t warpSize])
               (let ([t-from (shfl t it)])
                 (for/bounded ([my-i stride-x])
                   ;(pretty-display `(loop ,warp ,it ,t ,my-i)) ;; (loop 1 1 0 0)
                   (let ([global-x (+ offset-x (* it stride-x warpSize) (* stride-x t-from) my-i)]
                         [local-x (+ my-i (* it stride-x))])
                     (when (and (< global-x I-len)
                                (< local-x I-reg-len)
                                )
                       (vector-set! (vector-ref I-reg (+ t (* warp warpSize))) ;; thread in a block
                                    local-x ;; local index
                                    (vector-ref I global-x))
                       ;(pretty-display `(loop-true))
                   ))))))
           )))
     ]

    [(= (length blockDim) 2)
     (let* ([size-x (get-x sizes)]
            [size-y (get-y sizes)]
            [stride-x (get-x pattern)]
            [stride-y (get-y pattern)]
            [warp-shape-x (get-x warp-shape)]
            [warp-shape-y (get-y warp-shape)]
            [blockSize (apply * blockDim)]
            [iter-x (add1 (quotient (sub1 size-x) (* warp-shape-x stride-x)))]
            [iter-y (add1 (quotient (sub1 size-y) (* warp-shape-y stride-y)))]
            [I-len-x (vector-length (vector-ref I 0))]
            [I-len-y (vector-length I)]
            [I-reg-len-y (vector-length (vector-ref I-reg 0))]
            [I-reg-len-x (vector-length (vector-ref (vector-ref I-reg 0) 0))])
       (for ([warp (quotient blockSize warpSize)])
         ;(pretty-display `(>>> warp ,warp ,offset))
         (let ([offset-x (if (vector? offset)
                             (get-x (vector-ref offset (* warp warpSize)))
                             (vector-ref (get-x offset) (* warp warpSize)))]
               [offset-y (if (vector? offset)
                             (get-y (vector-ref offset (* warp warpSize)))
                             (vector-ref (get-y offset) (* warp warpSize)))])
           ;(pretty-display `(offset-x ,offset-x))
           (for/bounded ([it-y iter-y])
           (for/bounded ([it-x iter-x])
             ;(pretty-display `(iter ,warp ,it-y ,it-x))
             (for ([t warpSize])
               (let ([t-from (shfl t (+ (* it-y iter-x) it-x))])
               (for/bounded ([my-y stride-y])
               (for/bounded ([my-x stride-x])
                 ;(pretty-display `(loop ,warp ,it-x ,t ,my-x))
                 (let ([global-y (+ offset-y
                                    (* it-y warp-shape-y stride-y) ;; TODO (* size-y warp)
                                    (* (quotient t-from warp-shape-x) stride-y) my-y)]
                       [global-x (+ offset-x
                                    (* it-x warp-shape-x stride-x) ;; TODO (* size-x warp)
                                    (* (modulo t-from warp-shape-x) stride-x) my-x)]
                       [local-y (+ my-y (* it-y stride-y))]
                       [local-x (+ my-x (* it-x stride-x))]
                       )
                   ;(pretty-display `(info ,warp ,t ,global-y ,global-x ,local-y ,local-x))
                 (when (and (< global-y I-len-y) (< global-x I-len-x)
                            (< local-x I-reg-len-x) (< local-y I-reg-len-y)
                            )
                   (set I-reg local-x local-y
                        (+ t (* warp warpSize)) ;; thread in a block
                        (get I global-x global-y
                             ))))))))))
           )))
     ]

    ;; TODO
    [else (raise "unimplemented")]
    ))

;; Similar to global-to-local but
;; for storing I-reg in local memory/registers to I in global memory
(define (local-to-global I-reg I pattern offset sizes transpose
                         #:warp-shape [warp-shape warpSize]
                         #:round [round 1]
                         #:shfl [shfl (lambda (tid k) tid)]
                         #:size [gsize #f])
  (begin
    (if transpose
        (global-cost (reverse pattern) (reverse sizes))
        (global-cost pattern sizes))
  (assert (all? (@<= sizes (@* warp-shape pattern round)) true?))
  (assert (all? (@> sizes (@* warp-shape pattern (@- round 1))) true?))
  (cond
    [(= (length blockDim) 1)
     (let* ([size-x (get-x sizes)]
            [stride-x (get-x pattern)]
            [blockSize (apply * blockDim)]
            [iter-x (add1 (quotient (sub1 size-x) (* warpSize stride-x)))]
            [I-len (vector-length I)]
            [I-reg-len (vector-length I-reg)]
            [new-I-reg (make-vector blockSize #f)])
       ;(pretty-display `(iterate ,(quotient blockSize warpSize) ,iter-x ,stride-x))
       (for ([warp (quotient blockSize warpSize)])
         (let ([offset-x (if (vector? offset)
                             (get-x (vector-ref offset (* warp warpSize)))
                             (vector-ref (get-x offset) (* warp warpSize)))]
               [inc-x 0])
           ;(pretty-display `(offset-x ,offset-x))
           #;(for/bounded ([it iter-x])
             (for ([t warpSize])
               (for/bounded ([my-i stride-x])
                 (when (and (< inc-x size-x)
                            (< (+ offset-x inc-x) I-len)
                            (< (+ my-i (* it stride-x)) I-reg-len)
                            )
                   (vector-set! I (+ offset-x inc-x)
                                (vector-ref
                                 (vector-ref I-reg (+ t (* warp warpSize))) ;; thread in a block
                                 (+ my-i (* it stride-x)))) ;; local index
                   )
                 (set! inc-x (+ inc-x 1)))))
           (for/bounded ([it iter-x])
             (for ([t warpSize])
               (let ([t-from (shfl t it)])
               (for/bounded ([my-i stride-x])
                 ;(pretty-display `(loop ,warp ,it ,t ,my-i))
                 (when (and (< inc-x size-x)
                            (< (+ offset-x inc-x) I-len)
                            (< (+ my-i (* it stride-x)) I-reg-len)
                            )
                   (vector-set! I (+ offset-x (* it stride-x warpSize) (* stride-x t-from) my-i)
                                (vector-ref
                                 (vector-ref I-reg (+ t (* warp warpSize))) ;; thread in a block
                                 (+ my-i (* it stride-x)))) ;; local index
                   )))))
           )))
     ]

    ;; TODO
    [else (raise "unimplemented")]
    )))

;;;;;;;;;;;;;;;;;;;;;;;;;;; intra-warp shuffle operations ;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (shfl val lane)
  (define len (vector-length val))
  (define res (make-vector len #f))
  
  (define lane-vec
    (if (vector? lane)
        (for/vector ([i (vector-length lane)]) (modulo (get lane i) warpSize))
        (for/vector ([i len]) (modulo lane warpSize))))
  
  (for ([iter (quotient len warpSize)])
    (let ([offset (* iter warpSize)])
      (for ([i warpSize])
        (let ([i-dest (+ offset i)]
              [i-src (+ offset (get lane-vec (+ offset i)))])
        (set res i-dest (get val i-src))))))

  ;(set! cost (+ cost 2))
  res)

;; Scatter version of shuffle instruction. This instruction doesn't exist in GPU.
;; This function is for convenient uses.
(define (shfl-send val lane)
  (define len (vector-length val))
  (define res (make-vector len #f))
  
  (define lane-vec
    (if (vector? lane)
        (for/vector ([i (vector-length lane)]) (modulo (get lane i) warpSize))
        (for/vector ([i len]) (modulo lane warpSize))))
  
  (for ([iter (quotient len warpSize)])
    (let ([offset (* iter warpSize)])
      (for ([i warpSize])
        (let ([i-src (+ offset i)]
              [i-dest (+ offset (get lane-vec (+ offset i)))])
        (set res i-dest (get val i-src))))))

  ;(set! cost (+ cost 2))
  res)

;;;;;;;;;;;;;;;;;;;;;;;;;;; accumulators ;;;;;;;;;;;;;;;;;;;;;;;;;;;

(struct accumulator (val oplist opfinal veclen) #:mutable)

;; Multiset equal
(define (multiset= x y)
  (cond
    [(and (list? x) (list? y))
     (define ret (= (length x) (length y)))
     (for ([xi x])
       (let ([f (lambda (yi) (multiset= xi yi))])
         (set! ret (and ret (= (count f x) (count f y))))))
     ret]

    [else (equal? x y)]))

;; Accumulator equal
(define (acc=? x y recursive-equal?)
  (and (multiset= (accumulator-val x) (accumulator-val y))
       (equal? (accumulator-oplist x) (accumulator-oplist y))
       (equal? (accumulator-opfinal x) (accumulator-opfinal y))))

;; Create an accumulator or a vector of accumulators.
(define-syntax create-accumulator
  (syntax-rules ()
    ((create-accumulator op-list final-op)
     (accumulator (list) op-list final-op #f))
    ((create-accumulator op-list final-op blockDim)
     (build-vector (apply * blockDim)
                   (lambda (i) (accumulator (list) op-list final-op (apply * blockDim)))))))

(define-syntax-rule (get-accumulator-val x)
  (if (vector? x)
      (for/vector ([xi x]) (accumulator-val xi))
      (accumulator-val x)))

;; Convert to a vector of sorted lists.
(define (vector-of-list l veclen)
  (for/vector ([i veclen])
    (let ([each (map (lambda (x) (if (vector? x) (get x i) x)) l)])
      (%sort each (lambda (x y) (string<? (format "~a" x) (format "~a" y)))))))

(define (accumulate-merge x y)
  (cond
    [(and (accumulator? x) (accumulator? y))
     (accumulator (append (accumulator-val x) (accumulator-val y))
                  (accumulator-oplist x) (accumulator-opfinal x) (accumulator-veclen x))]

    [(accumulator? x) x]
    [(accumulator? y) y]
    [else (assert #f)]))

(define (accumulate-final x) x)

;; Accumulate val-list into an accumulator x or a vector of val-lists into a vector of accumulators.
(define (accumulate x val-list #:pred [pred #t])
  (define (f val-list op-list veclen)
    (if (= (length op-list) 1)
        (begin
          (assert (or (number? val-list) (vector? val-list)))
          (if (or (vector? val-list) (equal? veclen #f))
              val-list
              (for/vector ([i veclen]) val-list)))
        (let ([l (for/list ([val val-list])
                   (f val (cdr op-list) veclen))])
          (if veclen
              (vector-of-list l veclen)
              l))))

  (cond
    [(and (boolean? pred) (not pred)) (void)]
    
    [(vector? x)
     (define veclen (accumulator-veclen (get x 0)))
     (define addition (f val-list (accumulator-oplist (get x 0)) veclen))
     (define pred-vec (if (vector? pred) pred (for/vector ([i veclen]) pred)))
     ;(pretty-display `(pred-vec ,pred-vec ,(vector-length pred-vec)))
     (for ([i (vector-length x)])
       (let ([p (get pred-vec i)]
             [acc (get x i)]
             [add (get addition i)])
         (when p
           (set-accumulator-val! acc (cons add (accumulator-val acc))))))

     ;(unless (all? pred-vec false?)
     ;  (accumulate-cost (reverse (accumulator-oplist (vector-ref x 0))) addition))
     (set! cost (+ cost 1))
     ]

    [pred
     (define add (f val-list (accumulator-oplist x) #f))
     (set-accumulator-val! x (cons add (accumulator-val x)))
     
     ;(accumulate-cost (reverse (accumulator-oplist x)) add)
     (set! cost (+ cost 1)) 
     ])

  )

;; Check equivalence of x and y accumulators or vectors of accumulators.
(define (acc-equal? x y)
  (cond
    [(or (and (vector? x) (vector? y))
         (and (list? x) (list? y)))
     (define ret #t)
     (for ([xi x] [yi y])
       (set! ret (and ret (acc-equal? xi yi))))
     ret
     ]

    [(and (accumulator? x) (accumulator? y))
     (acc=? x y #t)]

    [else (equal? x y)]))

;; Print an accumulator or a vector of accumulators.
(define (acc-print x)
  (cond
    [(accumulator? x)
     `(accumulator ,(accumulator-val x))]
    [(vector? x)
     (define ret (for/vector ([xi x]) (acc-print xi)))
     (pretty-display ret)
     ]))

;;;;;;;;;;;;;;;;;;;;;;;;;;; run kernel ;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Get a warp id from a thread id.
(define (get-warpId threadID)
  (if (list? threadID)
      (let ([sum 0])
        (for ([id (reverse threadID)]
              [dim (cdr (reverse (cons 1 blockDim)))])
          (set! sum (+ sum (* id dim))))
        (quotient sum warpSize))
      (for/vector ([id threadID])
        (get-warpId id))))

;; Get a local thread id within a warp.
(define (get-idInWarp threadID)
  (if (list? threadID)
      (let ([sum 0])
        (for ([id (reverse threadID)]
              [dim (cdr (reverse (cons 1 blockDim)))])
          (set! sum (+ sum (* id dim))))
        (modulo sum warpSize)) ;; differ only here
      (for/vector ([id threadID])
        (get-idInWarp id))))

;; Get a global thread id.
(define (get-global-threadId threadId blockId)
  ;(pretty-display `(get-global-threadId ,threadId ,blockId ,blockDim))
  (if (list? threadId)
      (@++ threadId (@** blockId blockDim))
      (for/vector ([id threadId])
        (get-global-threadId id blockId))))

;; Get a vector of all thread ids given a threadblock size.
(define (get-threadId sizes)
  (define ret (list))
  (define (rec id sizes)
    (if (empty? sizes)
        (set! ret (cons id ret))
        (for ([i (car sizes)])
          (rec (cons i id) (cdr sizes)))))
  (rec (list) (reverse sizes))
  (list->vector (reverse ret)))

(define (run-grid kernel my-gridDim my-blockDim threadIds args)
  (set! gridDim my-gridDim)
  (set! blockDim my-blockDim)
  (set! blockSize (apply * my-blockDim))
  (reset-cost)
  
  (define (f blockID sizes)
    (if (empty? sizes)
        (begin
          (pretty-display `(blockID ,blockID ,blockDim ,threadIds))
          (apply kernel (append (list threadIds blockID blockDim) args)))
        (for ([i (car sizes)])
          (f (cons i blockID) (cdr sizes)))))
  (f (list) (reverse gridDim))
  ;;(pretty-display `(cost ,cost))
  )

;; Run a kernel.
(define-syntax-rule (run-kernel kernel my-blockDim my-gridDim x ...)
  (let ([Ids (get-threadId my-blockDim)])
    (run-grid kernel my-gridDim my-blockDim Ids (list x ...))))
