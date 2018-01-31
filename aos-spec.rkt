#lang rosette

(require "util.rkt" "cuda.rkt" "cuda-synth.rkt")

(define struct-size 3)
(define n-block 1)

(define (create-IO warpSize)
  (set-warpSize warpSize)
  (define block-size (* 1 warpSize))
  (define array-size (* n-block block-size))
  (define I-sizes (x-y-z (* array-size struct-size)))
  (define I (create-matrix I-sizes gen-uid))
  (define O (create-matrix I-sizes))
  (define O* (create-matrix I-sizes))
  (values block-size I-sizes I O O*))

(define (run-with-warp-size spec kernel w)
  (define-values (block-size I-sizes I O O*)
  (create-IO w))

  (define c (gcd struct-size warpSize))
  (define a (/ struct-size c))
  (define b (/ warpSize c))
  (pretty-display `(c ,c ,a ,b))

  (run-kernel spec (x-y-z block-size) (x-y-z n-block) I O a b c)
  (run-kernel kernel (x-y-z block-size) (x-y-z n-block) I O* a b c)
  (define ret (equal? O O*))
  (pretty-display `(O ,O))
  (pretty-display `(O* ,O*))
  ret)

(define (AOS-load-spec threadId blockID blockDim I O a b c)
  (define I-cached (create-matrix (x-y-z struct-size)))
  (define warpID (get-warpId threadId))
  (define offset (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  (define gid (get-global-threadId threadId blockID))
  (global-to-warp-reg I I-cached
                 (x-y-z struct-size)
                 offset (x-y-z (* warpSize struct-size)) #f)
  (warp-reg-to-global I-cached O
                      (x-y-z 1) offset (x-y-z (* warpSize struct-size)) #f)
  )


(define (AOS-load-sketch threadId blockID blockDim I O a b c)
  (define I-cached (create-matrix (x-y-z struct-size)))
  (define O-cached (for/vector ([i blockSize]) (create-matrix (x-y-z struct-size))))
  (define warpID (get-warpId threadId))
  (define offset (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  (define gid (get-global-threadId threadId blockID))
  (global-to-warp-reg I I-cached
                 (x-y-z 1)
                 offset
                 (x-y-z (* warpSize struct-size)) #f)

  (define j (get-idInWarp threadId))
  (for/bounded ([i struct-size])
    (let* ([p (modulo (- i j) struct-size)]
           [b-inv 1]
           [q1 (modulo (* b-inv (quotient (+ (- c 1) p) c)) a)]
           [q2 (* (modulo (* (- c 1) p) c) a)]
           [index (modulo (+ q1 q2) struct-size)] ;; step 1 (column shuffle): index = s-inv(i)
           [lane (+ (modulo (+ i (quotient j b)) struct-size) (* j struct-size))]  ;; step 2 (row shuffle): lane = d'(j)
           [x (shfl (get I-cached index) lane)]
           [index-o (modulo (+ i (quotient j b)) struct-size)] ;; step 3 (column rotate): index-o = r(i)
           )
      (pretty-display `(index-i ,index))
      (pretty-display `(lane ,lane))
      (pretty-display `(x ,x))
      (pretty-display `(index-o ,index-o))
      (newline)
      (set O-cached index-o x))
      )
  
  (warp-reg-to-global O-cached O
                      (x-y-z 1)
                      offset
                      (x-y-z (* warpSize struct-size)) #f)
  )

(define (AOS-load-sketch2 threadId blockID blockDim I O a b c)
  #|
  (define log-a (bvlog a))
  (define log-b (bvlog b))
  (define log-c (bvlog c))
  (define log-m (bvlog struct-size))
  (define log-n (bvlog warpSize))
|#
  
  (define I-cached (create-matrix (x-y-z struct-size)))
  (define O-cached (for/vector ([i blockSize]) (create-matrix (x-y-z struct-size))))
  (define warpID (get-warpId threadId))
  (define offset (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  (define gid (get-global-threadId threadId blockID))
  (global-to-warp-reg I I-cached
                 (x-y-z 1)
                 offset
                 (x-y-z (* warpSize struct-size)) #f)

  (define j (get-idInWarp threadId))
  (for/bounded ([i struct-size])
    (let* ([index (modulo (* (- c 1) (- i j)) struct-size)] ;; step 1 (column shuffle): index = s-inv(i)
           [lane (+ (modulo (+ i (quotient j b)) struct-size) (* j struct-size))]  ;; step 2 (row shuffle): lane = d'(j)
           
           #;[index (@int (extract (bvsub (bvshl (bvsub (@bv i) (@bv j)) log-c)
                                        (bvsub (@bv i) (@bv j))) log-m))]
           #;[lane (@int (bvadd (extract (bvadd (@bv i) (bvlshr (@bv j) log-b)) log-m)
                              (bvshl (@bv j) log-m)))]
           [x (shfl (get I-cached index) lane)]
           [index-o (modulo (+ i (quotient j b)) struct-size)] ;; step 3 (column rotate): index-o = r(i)
           ;[index-o (@int (extract (bvadd (@bv i) (bvlshr (@bv j) log-b)) log-m))]
           )
      ;(pretty-display `(index-i ,index))
      ;(pretty-display `(lane ,lane))
      ;(pretty-display `(x ,x))
      ;(pretty-display `(index-o ,index-o))
      ;(newline)
      ;(unique-warp (modulo lane warpSize))
      (set O-cached index-o x))
      )
  
  (warp-reg-to-global O-cached O
                      (x-y-z 1)
                      offset
                      (x-y-z (* warpSize struct-size)) #f)
  )

(define (test)
  (for ([w (list 32)])
    (let ([ret (run-with-warp-size AOS-load-spec AOS-load-sketch2 w)])
      (pretty-display `(test ,w ,ret))))
  )
(test)