#lang rosette

(require "util.rkt" "cuda.rkt" "cuda-synth.rkt")

(define struct-size 4)
(define n-block 1)

(define (create-IO warpSize)
  (set-warpSize warpSize)
  (define block-size (* 2 warpSize))
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

  (run-kernel spec (x-y-z block-size) (x-y-z n-block) I O a b c)
  (run-kernel kernel (x-y-z block-size) (x-y-z n-block) I O* a b c)
  (define ret (equal? O O*))
  ;(pretty-display `(O ,O))
  ;(pretty-display `(O* ,O*))
  ret
  )

(define (AOS-load-spec threadId blockID blockDim I O a b c)
  (define I-cached (create-matrix (x-y-z struct-size)))
  (define warpID (get-warpId threadId))
  (define offset (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  (define gid (get-global-threadId threadId blockID))
  (global-to-warp-reg I I-cached
                 (x-y-z 1)
                 offset (x-y-z (* warpSize struct-size)) #f)
  (warp-reg-to-global I-cached O
                      (x-y-z struct-size) offset (x-y-z (* warpSize struct-size)) #f)
  )

;; cpu time: 372563 real time: 3828661
(define (AOS-load-test3 threadId blockID blockDim I O a b c)
   (define I-cached (create-matrix (x-y-z struct-size)))
   (define O-cached
     (for/vector ((i blockSize)) (create-matrix (x-y-z struct-size))))
   (define warpID (get-warpId threadId))
   (define offset
     (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))
   (define gid (get-global-threadId threadId blockID))
   (global-to-warp-reg
    I
    I-cached
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f)
   (define indices (make-vector struct-size))
   (define indices-o (make-vector struct-size))
   (define localId (get-idInWarp threadId))
   (for/bounded
    ((i struct-size))
    (let* ((index (modulo (+ (- (@dup i) localId) localId) struct-size))
           (lane
            (+
             (-
              (* (* localId (@dup a)) (@dup a))
              (* (* (@dup i) (@dup a)) (@dup struct-size)))
             (+
              (- (+ localId localId) (- (@dup i) (@dup 1)))
              (- (- (@dup i) (@dup 1)) (+ (@dup i) (@dup i))))))
           (x (shfl (get I-cached index) lane))
           (index-o
            (modulo
             (-
              (+ (* localId (@dup a)) (- localId (@dup i)))
              (* (+ localId (@dup i)) (@dup a)))
             struct-size)))
      (set O-cached index-o x)))
   (warp-reg-to-global
    O-cached
    O
    (x-y-z 1)
    offset
    (x-y-z (* warpSize struct-size))
    #f))

;; warpSize 32, depth 2/4/3, constraint col+row permute, distinct?: 
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

  (define indices (make-vector struct-size))
  (define indices-o (make-vector struct-size))
  (define localId (get-idInWarp threadId))
  (for/bounded ([i struct-size])
    (let* ([index (modulo (?index localId (@dup i) [a b c struct-size warpSize] 2) struct-size)]  ; (?index localId (@dup i) 1)
           [lane (?lane localId (@dup i) [a b c struct-size warpSize] 4)]  ; (+ (modulo (+ i (quotient localId 2)) 2) (* localId 2))
           [x (shfl (get I-cached index) lane)]
           [index-o (modulo (?index localId (@dup i) [a b c struct-size warpSize] 3) struct-size)])
      ;(unique-warp (modulo lane warpSize))
      ;(vector-set! indices i index)
      ;(vector-set! indices-o i index-o)
      (set O-cached index-o x))
      )
  #;(for ([t blockSize])
    (let ([l (for/list ([i struct-size]) (vector-ref (vector-ref indices i) t))]
          [lo (for/list ([i struct-size]) (vector-ref (vector-ref indices-o i) t))])
      (unique-list l)
      (unique-list lo)))
  
  (warp-reg-to-global O-cached O
                      (x-y-z 1)
                      offset
                      (x-y-z (* warpSize struct-size)) #f)
  )

(define (test)
  (for ([w (list 32)])
    (let ([ret (run-with-warp-size AOS-load-spec AOS-load-test3 w)])
      (pretty-display `(test ,w ,ret))))
  )
;(test)

;; struct-size = 3, no permutation constraint: 44/1058 s
;; struct-size = 3, row+col permutation, distinct?: 49/1118 s

;; struct-size = 2: 20/90
;; struct-size = 4: 56/994
(define (synthesis)
  (pretty-display "solving...")
  (define sol (time (solve (assert (andmap (lambda (w) (run-with-warp-size AOS-load-spec AOS-load-sketch w))
                                           (list 32))))))
  (print-forms sol)
  )
(synthesis)
