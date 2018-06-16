#lang rosette

(require "util.rkt" "cuda.rkt" "cuda-synth.rkt")

(define struct-size 6)
(define n-block 2)

(define (create-IO warpSize)
  (set-warpSize warpSize)
  (define block-size (* 2 warpSize))
  (define array-size (* n-block block-size))
  (define I-sizes (x-y-z (* array-size struct-size)))
  (define O-sizes (x-y-z array-size))
  (define I (create-matrix I-sizes gen-uid))
  (define O (create-matrix O-sizes))
  (define O* (create-matrix O-sizes))
  (values block-size I-sizes O-sizes I O O*))

(define (run-with-warp-size spec kernel w)
  (define-values (block-size I-sizes O-sizes I O O*)
    (create-IO w))
  
  (define c (gcd struct-size warpSize))
  (define a (/ struct-size c))
  (define b (/ warpSize c))

  (reset-cost)
  (spec I O O-sizes)
  (pretty-display `(spec-cost ,(get-cost)))
  (run-kernel kernel (x-y-z block-size) (x-y-z n-block) I O* I-sizes O-sizes a b c)
  ;(acc-print O)
  ;(acc-print O*)
  (acc-equal? O O*))

(define (AOS-sum-spec I O O-sizes)
  (for ([i (get-x O-sizes)])
    (let ([o (create-accumulator o (list +) identity)])
      (for ([j struct-size])
        (accumulate o (get I (+ (* i struct-size) j))))
      (set O i o)))
  )

(define (AOS-sum-slow threadId blockID blockDim I O I-sizes O-sizes a b c)
  (define I-cached (create-matrix (x-y-z struct-size)))
  (define warpID (get-warpId threadId))
  (define offset (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  (define gid (get-global-threadId threadId blockID))
  (global-to-warp-reg I I-cached
                 (x-y-z struct-size)
                 offset (x-y-z (* warpSize struct-size)) #f)

  (define localId (get-idInWarp threadId))
  (define o (create-accumulator o (list +) identity blockDim))
  (for ([i struct-size])
    (let* ([index (@dup i)]
           [lane localId]
           [x (shfl (get I-cached index) lane)])
      (accumulate o x)
      ))
  (reg-to-global o O gid)
  )

(define (print-line v n)
  (define i 0)
  (for ([x v])
    (display (format "~a " x))
    (set! i (add1 i))
    (when (= i n)
      (set! i 0)
      (newline))))

(define (AOS-sum6-trove threadId blockID blockDim I O I-sizes O-sizes a b c)
  
  (define I-cached (create-matrix (x-y-z struct-size)))
  (define warpID (get-warpId threadId))
  (define offset (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  (define gid (get-global-threadId threadId blockID))
  (global-to-warp-reg I I-cached
                        (x-y-z 2) ;; stride
                        ;(x-y-z (?warp-offset [(get-x blockID) (get-x blockDim)] [warpID warpSize])) ;; offset
                        (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize))
                        ;(x-y-z (?warp-size warpSize 1)) ;; load size
                        (x-y-z (* warpSize struct-size))
                        #f)

  (define localId (get-idInWarp threadId))
  (define o (create-accumulator o (list +) identity blockDim))
  (define indices (make-vector struct-size))
  
  (reset-cost)


  (define r20 (modulo (* 3 localId) warpSize))
  (define r22 (modulo (+ 2 (* 3 localId)) warpSize))
  (define r24 (modulo (+ 1 (* 3 localId)) warpSize))

  (define r27 (- localId (* 3 (quotient localId 3))))
  (define p2 (= (modulo r27 2) 0))
  (define p3 (= (modulo (quotient r27 2) 2) 0))

  (define r28 (modulo localId 3))

  (define idx1 (create-matrix (x-y-z struct-size blockSize)))
  (define idx2 (create-matrix (x-y-z struct-size blockSize)))
  (for ([i struct-size])
    (set idx1 (@dup i) (ite p2
                            (get I-cached (@dup i))
                            (get I-cached (@dup (modulo (+ i 2) struct-size)))))
    )
  (for ([i struct-size])
    (set idx2 (@dup i) (ite p3
                            (get idx1 (@dup i))
                            (get idx1 (@dup (modulo (- i 2) struct-size)))))
    )

  (pretty-display ">>> r20")
  (print-line r20 32)
  (pretty-display ">>> r22")
  (print-line r22 32)
  (pretty-display ">>> r24")
  (print-line r24 32)
  (accumulate o (shfl (get idx2 (@dup 0)) r20) #:pred (@dup #t))
  (accumulate o (shfl (get idx2 (@dup 1)) r20) #:pred (@dup #t))
  (accumulate o (shfl (get idx2 (@dup 2)) r22) #:pred (@dup #t))
  (accumulate o (shfl (get idx2 (@dup 3)) r22) #:pred (@dup #t))
  (accumulate o (shfl (get idx2 (@dup 4)) r24) #:pred (@dup #t))
  (accumulate o (shfl (get idx2 (@dup 5)) r24) #:pred (@dup #t))

  (reg-to-global o O gid)
  )


(define (AOS-sum8-trove threadId blockID blockDim I O I-sizes O-sizes a b c)
  
  (define I-cached (create-matrix (x-y-z struct-size)))
  (define warpID (get-warpId threadId))
  (define offset (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  (define gid (get-global-threadId threadId blockID))
  (global-to-warp-reg I I-cached
                        (x-y-z 4) ;; stride
                        ;(x-y-z (?warp-offset [(get-x blockID) (get-x blockDim)] [warpID warpSize])) ;; offset
                        (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize))
                        ;(x-y-z (?warp-size warpSize 1)) ;; load size
                        (x-y-z (* warpSize struct-size))
                        #f)

  (define localId (get-idInWarp threadId))
  (define o (create-accumulator o (list +) identity blockDim))
  (define indices (make-vector struct-size))
  
  (reset-cost)

  (define r24 (modulo (quotient localId 16) 2))
  (define r26 (modulo (+ (* 2 localId) r24) warpSize))
  (define p1 (= r24 0))
  (define r29 (ite p1 (+ r26 2) r26))
  (define r30 (- r29 1))

  (define p3 (= (modulo localId 2) 0))

  (define idx1 (create-matrix (x-y-z struct-size blockSize)))
  (for ([i struct-size])
    (set idx1 (@dup i) (ite p3
                            (get I-cached (@dup i))
                            (get I-cached (@dup (modulo (+ i 4) struct-size)))))
    ;(accumulate o (get idx1 (@dup i)))
    )

  (accumulate o (shfl (get idx1 (@dup 0)) r26) #:pred (@dup #t))
  (accumulate o (shfl (get idx1 (@dup 1)) r26) #:pred (@dup #t))
  (accumulate o (shfl (get idx1 (@dup 2)) r26) #:pred (@dup #t))
  (accumulate o (shfl (get idx1 (@dup 3)) r26) #:pred (@dup #t))
  (accumulate o (shfl (get idx1 (@dup 4)) r30) #:pred (@dup #t))
  (accumulate o (shfl (get idx1 (@dup 5)) r30) #:pred (@dup #t))
  (accumulate o (shfl (get idx1 (@dup 6)) r30) #:pred (@dup #t))
  (accumulate o (shfl (get idx1 (@dup 7)) r30) #:pred (@dup #t))

  (reg-to-global o O gid)
  )

(define (AOS-sum5 threadId blockID blockDim I O I-sizes O-sizes a b c)
   (define I-cached (create-matrix (x-y-z struct-size)))
   (define warpID (get-warpId threadId))
   (define offset
     (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))
   (define gid (get-global-threadId threadId blockID))
   (global-to-warp-reg
    I
    I-cached
    (x-y-z 1)
    (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize))
    (x-y-z (* warpSize struct-size))
    #f)
   (define localId (get-idInWarp threadId))
   (define o (create-accumulator o (list +) identity blockDim))
   (define indices (make-vector struct-size))
   (reset-cost)
   (define r0 (modulo (+ (+ (* localId a) 0) 2) warpSize))
   (define r20
     (modulo (+ (+ (* localId a) 0) (+ (* r0 -1) (quotient r0 1)) b) warpSize))
   (define r21
     (modulo
      (+
       (+ (* localId -1) (quotient localId c))
       (+ (* r0 0) (quotient r0 1))
       0)
      b))
   (define r22
     (modulo
      (+
       (+ (* localId -1) (quotient localId 1))
       (+ (* r0 0) (quotient r0 1))
       2)
      warpSize))
   (define r23
     (modulo (+ (+ (* localId 0) 0) (+ (* r0 0) (quotient r0 1)) -1) warpSize))
   (define r24
     (modulo
      (+
       (+ (* localId -1) (quotient localId c))
       (+ (* r0 0) (quotient r0 1))
       1)
      warpSize))
   (define r28
     (modulo (+ (+ (* localId c) (quotient localId c)) 0) struct-size))
   (define p1 (= (modulo r28 2) 0))
   (define p2 (= (modulo (quotient r28 2) 2) 0))
   (define p3 (= (modulo (quotient r28 4) 2) 0))
   (define idx0 (create-matrix (x-y-z struct-size blockSize)))
   (define idx1 (create-matrix (x-y-z struct-size blockSize)))
   (define idx2 (create-matrix (x-y-z struct-size blockSize)))
   (for
    ((i struct-size))
    (set
     idx0
     (@dup i)
     (ite
      p1
      (get I-cached (@dup i))
      (get I-cached (@dup (modulo (+ i 1) struct-size))))))
   (for
    ((i struct-size))
    (set
     idx1
     (@dup i)
     (ite
      p2
      (get idx0 (@dup i))
      (get idx0 (@dup (modulo (+ i 2) struct-size))))))
   (for
    ((i struct-size))
    (set
     idx2
     (@dup i)
     (ite
      p3
      (get idx1 (@dup i))
      (get idx1 (@dup (modulo (+ i 4) struct-size))))))
   (accumulate o (shfl (get idx2 (@dup 0)) r20) #:pred (@dup #t))
   (accumulate o (shfl (get idx2 (@dup 1)) r21) #:pred (@dup #t))
   (accumulate o (shfl (get idx2 (@dup 2)) r22) #:pred (@dup #t))
   (accumulate o (shfl (get idx2 (@dup 3)) r23) #:pred (@dup #t))
   (accumulate o (shfl (get idx2 (@dup 4)) r24) #:pred (@dup #t))
   (reg-to-global o O gid))

(define (AOS-sum7 threadId blockID blockDim I O I-sizes O-sizes a b c)
   (define I-cached (create-matrix (x-y-z struct-size)))
   (define warpID (get-warpId threadId))
   (define offset
     (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))
   (define gid (get-global-threadId threadId blockID))
   (global-to-warp-reg
    I
    I-cached
    (x-y-z 1)
    (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize))
    (x-y-z (* warpSize struct-size))
    #f)
   (define localId (get-idInWarp threadId))
   (define o (create-accumulator o (list +) identity blockDim))
   (define indices (make-vector struct-size))
   (reset-cost)
   (define r0
     (modulo (+ (+ (* localId -1) (quotient localId 1)) b) struct-size))
   (define r20
     (modulo (+ (+ (* localId a) 0) (+ (* r0 -1) (quotient r0 1)) b) warpSize))
   (define r21
     (modulo (+ (+ (* localId a) 0) (+ (* r0 0) (quotient r0 c)) 0) warpSize))
   (define r22
     (modulo
      (+ (+ (* localId struct-size) 0) (+ (* r0 -1) (quotient r0 c)) c)
      warpSize))
   (define r23
     (modulo (+ (+ (* localId a) 0) (+ (* r0 0) (quotient r0 1)) c) warpSize))
   (define r24
     (modulo (+ (+ (* localId a) 0) (+ (* r0 -1) (quotient r0 1)) 2) warpSize))
   (define r25
     (modulo
      (+ (+ (* localId struct-size) 0) (+ (* r0 0) (quotient r0 c)) 2)
      warpSize))
   (define r26
     (modulo
      (+ (+ (* localId struct-size) 0) (+ (* r0 0) (quotient r0 1)) -1)
      warpSize))
   (define r28 (modulo (+ (+ (* localId -1) (quotient localId -1)) 0) a))
   (define p1 (= (modulo r28 2) 0))
   (define p2 (= (modulo (quotient r28 2) 2) 0))
   (define p3 (= (modulo (quotient r28 4) 2) 0))
   (define idx0 (create-matrix (x-y-z struct-size blockSize)))
   (define idx1 (create-matrix (x-y-z struct-size blockSize)))
   (define idx2 (create-matrix (x-y-z struct-size blockSize)))
   (for
    ((i struct-size))
    (set
     idx0
     (@dup i)
     (ite
      p1
      (get I-cached (@dup i))
      (get I-cached (@dup (modulo (+ i 1) struct-size))))))
   (for
    ((i struct-size))
    (set
     idx1
     (@dup i)
     (ite
      p2
      (get idx0 (@dup i))
      (get idx0 (@dup (modulo (+ i 2) struct-size))))))
   (for
    ((i struct-size))
    (set
     idx2
     (@dup i)
     (ite
      p3
      (get idx1 (@dup i))
      (get idx1 (@dup (modulo (+ i 4) struct-size))))))
   (accumulate o (shfl (get idx2 (@dup 0)) r20) #:pred (@dup #t))
   (accumulate o (shfl (get idx2 (@dup 1)) r21) #:pred (@dup #t))
   (accumulate o (shfl (get idx2 (@dup 2)) r22) #:pred (@dup #t))
   (accumulate o (shfl (get idx2 (@dup 3)) r23) #:pred (@dup #t))
   (accumulate o (shfl (get idx2 (@dup 4)) r24) #:pred (@dup #t))
   (accumulate o (shfl (get idx2 (@dup 5)) r25) #:pred (@dup #t))
   (accumulate o (shfl (get idx2 (@dup 6)) r26) #:pred (@dup #t))
   (reg-to-global o O gid))

(define (test)
  (for ([w (list 32)])
    (let ([ret (run-with-warp-size AOS-sum-spec AOS-sum6-trove w)])
      (pretty-display `(test ,w ,ret ,(get-cost)))))
  )
(test)

(define (synthesis)
  (pretty-display "solving...")
  (assert
   (andmap (lambda (w) (run-with-warp-size AOS-sum-spec AOS-sum8-trove w))
           (list 32)))
  (define cost (get-cost))
  (define sol (time (optimize #:minimize (list cost) #:guarantee (assert #t))))

  (define this-cost (evaluate cost sol))
  (print-forms sol)
  (pretty-display `(cost ,this-cost))
  
  ;(define sol2 (solve (assert (< cost this-cost))))
  ;(pretty-display `(cost2 ,(evaluate cost sol2)))
  )
;(synthesis)

(define (load-synth)
  (define-values (block-size I-sizes O-sizes I O O*)
    (create-IO 4))
  
  ;; Store
  (define (AOS-sum-store threadId blockId blockDim O)
    (define warpID (get-warpId threadId))
    (define o
      (for/vector ([w  warpID]
                   [t threadId])
        (ID t w blockId)))
    (reg-to-global o O (get-global-threadId threadId blockId))
    )
  
  ;; Run spec
  (AOS-sum-spec I O O-sizes)
  
  ;; Collect IDs
  (define IDs (create-matrix O-sizes))
  (run-kernel AOS-sum-store (x-y-z block-size) (x-y-z n-block) IDs)
  (define-values (threads warps blocks) (get-grid-storage))
  (collect-inputs O IDs threads warps blocks)
  (define n-regs (num-regs warps I))
  (pretty-display `(n-regs ,n-regs))

  ;; Load
  (define (AOS-sum-load threadId blockId blockDim I warp-input-spec)
    (define warpId (get-warpId threadId))
    ;; sketch starts
    (define I-cached (create-matrix (x-y-z n-regs)))
    (global-to-warp-reg I I-cached
                        (x-y-z 1) ;; stride
                        (x-y-z (?warp-offset [(get-x blockId) (get-x blockDim)] [warpId warpSize])) ;; offset
                        (x-y-z (?warp-size warpSize 1)) ;; load size
                        #f)
    ;; sketch ends
    (check-warp-input warp-input-spec I I-cached warpId blockId)
    )

  (run-kernel AOS-sum-load (x-y-z block-size) (x-y-z n-block) I warps)
  (define sol (time (solve (assert #t))))
    #;(time
     (synthesize
      #:forall (symbolics I)
      #:guarantee (assert #t)))
  (when (sat? sol)
    (print-forms sol))
  )
;(load-synth)

