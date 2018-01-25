#lang rosette

(require "util.rkt" "cuda.rkt" "cuda-synth.rkt")

(define struct-size 3)
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

  (spec I O O-sizes)
  (run-kernel kernel (x-y-z block-size) (x-y-z n-block) I O* I-sizes O-sizes a b c)
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

(define (AOS-sum-fast threadId blockID blockDim I O I-sizes O-sizes a b c)
  (define log-a (bvlog a))
  (define log-b (bvlog b))
  (define log-c (bvlog c))
  (define log-m (bvlog struct-size))
  (define log-n (bvlog warpSize))
  
  (define I-cached (create-matrix (x-y-z struct-size)))
  (define warpID (get-warpId threadId))
  (define offset (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  (define gid (get-global-threadId threadId blockID))
  (global-to-warp-reg I I-cached
                 (x-y-z 1)
                 offset (x-y-z (* warpSize struct-size)) #f)
  ;(pretty-display `(I-cached ,I-cached))

  (define localId (get-idInWarp threadId))
  (define o (create-accumulator o (list +) identity blockDim))
  (define indices (make-vector struct-size))
  (for ([i struct-size])
    (let* ([index (modulo (+ i localId) struct-size)]
           ;[index (@int (extract (bvadd (@bv i) (@bv localId)) log-m))]
           [lane (+ (modulo (+ i (quotient localId b)) struct-size) (* localId struct-size))]
           ;[lane (@int (bvadd (extract (bvadd (@bv i) (bvlshr (@bv localId) log-b)) log-m)
           ;                   (bvshl (@bv localId) log-m)))]
           [x (shfl (get I-cached index) lane)])
      ;(pretty-display `(lane ,lane))
      (vector-set! indices i index)
      (unique-warp (modulo lane warpSize))
      (accumulate o x)
      ))
  (pretty-display `(indices ,indices))
  (for ([t blockSize])
    (let ([l (for/vector ([i struct-size])
               (vector-ref (vector-ref indices i) t))])
      (unique l)))
  (reg-to-global o O gid)
  )

(define (AOS-sum-test2 threadId blockID blockDim I O I-sizes O-sizes a b c)
   (define log-a (bvlog a))
   (define log-b (bvlog b))
   (define log-c (bvlog c))
   (define log-m (bvlog struct-size))
   (define log-n (bvlog warpSize))
   (define I-cached (create-matrix (x-y-z struct-size)))
   (define warpID (get-warpId threadId))
   (define offset
     (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))
   (define gid (get-global-threadId threadId blockID))
   (global-to-warp-reg
    I
    I-cached
    (x-y-z 1)
    (x-y-z (+ 0 (* 2 (get-x blockID) (get-x blockDim)) (* 2 warpID warpSize)))
    (x-y-z (+ (?warp-size warpSize (- 1 1)) warpSize))
    #f)
   (define localId (get-idInWarp threadId))
   (define o (create-accumulator o (list +) identity blockDim))
   (define indices (make-vector struct-size))
   (for/bounded
    ((i struct-size))
    (let* ((index
            (@int
             (extract
              (bvlshr (bvsub (@bv localId) (@dup (@bv i))) log-a)
              log-m)))
           (lane
            (@int
             (bvadd
              (extract
               (bvadd
                (extract (@dup (@bv i)) log-n)
                (bvlshr (@bv localId) log-b))
               log-m)
              (bvlshr (bvshl (@bv localId) log-n) log-b))))
           (x (shfl (get I-cached index) lane)))
      ;(unique-warp (modulo lane warpSize))
      (vector-set! indices i index)
      (accumulate o x #:pred (@dup #t))))
  #;(for ([t blockSize])
    (let ([l (for/vector ([i struct-size])
               (vector-ref (vector-ref indices i) t))])
      (unique l)))
   (reg-to-global o O gid))

(define (AOS-sum-test3 threadId blockID blockDim I O I-sizes O-sizes a b c)
   (define I-cached (create-matrix (x-y-z struct-size)))
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
   (define localId (get-idInWarp threadId))
   (define o (create-accumulator o (list +) identity blockDim))
   (for/bounded
    ((i 3))
    (let* ((index
            (modulo
             (- (* (@dup i) (@dup warpSize)) (* localId (@dup warpSize)))
             struct-size))
           (lane
            (+
             (-
              (+ (quotient (@dup i) (@dup warpSize)) (@dup i))
              (-
               (quotient localId (@dup c))
               (quotient (@dup i) (@dup warpSize))))
             (+
              (+ (* (@dup i) (@dup b)) (* localId (@dup struct-size)))
              (+ (* localId (@dup warpSize)) (* localId (@dup c))))))
           (x (shfl (get I-cached index) lane)))
      (accumulate o x #:pred (= localId localId))))
   (reg-to-global o O gid))


;; fix loop bound
;; m = 2, warpsize = 4 8: 16/62 s ***
;; m = 2, warpsize = 4 8, ref: 19/98 s
;; m = 2, warpsize = 4 8, constraint row permute: 12/127 s
;; m = 2, warpsize = 4 8, constraint row permute, distinct?: 13/64 s
;; m = 2, warpsize = 4 8, constraint col+row permute, distinct?: 10/62 s ***
;; m = 2, warpsize = 4 8, constraint col+row permute, distinct?, ref: 17/82  s

;; m = 2, warpsize = 4 8, bv: 7/39 s
;; m = 2, warpsize = 4 8, bv, constraint row permute: 15/26 s
;; m = 2, warpsize = 4 8, bv, constraint row permute, distinct?: 7/39 s
;; m = 2, warpsize = 4 8, bv, constraint col+row permute: 7/39 s
;; m = 2, warpsize = 4 8, bv, constraint col+row permute, distinct?: 7/39 s

;; m = 2, warpsize = 4 8, bv, ref: 8/45 s
;; m = 2, warpsize = 4 8, bv, constraint row permute, ref: 9/27 s
;; m = 2, warpsize = 4 8, bv, constraint row permute, distinct?, ref: 27/53 s
;; m = 2, warpsize = 4 8, bv, constraint col+row permute, ref: 15/27 s

;; m = 3, warpsize = 32: error
;; m = 3, warpsize = 32, constraint col+row permute, distinct?: error

(define (AOS-sum-sketch threadId blockID blockDim I O I-sizes O-sizes a b c)
  #|
  (define log-a (bvlog a))
  (define log-b (bvlog b))
  (define log-c (bvlog c))
  (define log-m (bvlog struct-size))
  (define log-n (bvlog warpSize))
|#
  
  (define I-cached (create-matrix (x-y-z struct-size)))
  (define warpID (get-warpId threadId))
  (define offset (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize)))  ;; warpID = (threadIdy * blockDimx + threadIdx)/warpSize
  (define gid (get-global-threadId threadId blockID))
  (global-to-warp-reg I I-cached
                        (x-y-z 1) ;; stride
                        (x-y-z (?warp-offset [(get-x blockID) (get-x blockDim)] [warpID warpSize])) ;; offset
                        (x-y-z (?warp-size warpSize 1)) ;; load size
                        #f)

  (define localId (get-idInWarp threadId))
  (define o (create-accumulator o (list +) identity blockDim))
  (define indices (make-vector struct-size))
  (for/bounded ([i struct-size])
    (let* ([index (modulo (?index localId (@dup i) [a b c struct-size warpSize] 2) struct-size)]
           [lane (?lane localId (@dup i) [a b c struct-size warpSize] 4)]
           ;[index (@int (extract (?lane-log (@bv localId) (@dup (@bv i)) [log-a log-b log-c log-m log-n] 2) log-m))]
           ;[lane (@int (?lane-log (@bv localId) (@dup (@bv i)) [log-a log-b log-c log-m log-n] 4))]
           [x (shfl (get I-cached index) lane)])
      (unique-warp (modulo lane warpSize))
      ;(vector-set! indices i index)
      (accumulate o x #:pred (?cond localId (@dup i)))
      ))
  #;(for ([t blockSize])
    (let ([l (for/list ([i struct-size])
               (vector-ref (vector-ref indices i) t))])
      (unique-list l)))
  
  (reg-to-global o O gid)
  )



(define (test)
  (for ([w (list 32)])
    (let ([ret (run-with-warp-size AOS-sum-spec AOS-sum-test3 w)])
      (pretty-display `(test ,w ,ret))))
  )
;(test)

(define (synthesis)
  (pretty-display "solving...")
  (define sol (time (solve (assert (andmap (lambda (w) (run-with-warp-size AOS-sum-spec AOS-sum-sketch w))
                                     (list 4 8))))))
  (print-forms sol)
  )
(synthesis)

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