#lang rosette

(require "util.rkt" "cuda.rkt" "cuda-synth.rkt")

(define struct-size 4)
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
                 ;;(x-y-z (?warp-offset [(get-x blockID) (get-x blockDim)] [warpID warpSize]))
                 offset
                 (x-y-z (* warpSize struct-size)) #f)

  (define localId (get-idInWarp threadId))
  (for/bounded ([i struct-size])
    (let* ([p (modulo (- i localId) struct-size)]
           [b-inv 1]
           [q1 (modulo (* b-inv (quotient (+ (- c 1) p) c)) a)]
           [index (modulo (+ q1 (* (modulo (* (- c 1) p) c) a)) struct-size)]
           ;[q1 (modulo (quotient (+ (- c 1) i) c) a)]
           ;[q (+ q1 (* (modulo (* (- c 1) i) c) a))]
           ;[index (modulo (- q localId) struct-size)]  ; (?index localId (@dup i) 1)
           [lane (+ (modulo (+ i (quotient localId b)) struct-size) (* localId struct-size))]  ; (+ (modulo (+ i (quotient localId 2)) 2) (* localId 2))
           [x (shfl (get I-cached index) lane)]
           [index-o (modulo (+ i (quotient localId b)) struct-size)])
      (pretty-display `(index-i ,index))
      (pretty-display `(lane ,lane))
      (pretty-display `(x ,x))
      (pretty-display `(index-o ,index-o))
      (newline)
      (set O-cached index-o x))
      )
  
  (warp-reg-to-global O-cached O
                      (x-y-z 1)
                      ;;(x-y-z (?warp-offset [(get-x blockID) (get-x blockDim)] [warpID warpSize]))
                      offset
                      (x-y-z (* warpSize struct-size)) #f)
  )

(define (test)
  (for ([w (list 32)])
    (let ([ret (run-with-warp-size AOS-load-spec AOS-load-sketch w)])
      (pretty-display `(test ,w ,ret))))
  )
(test)