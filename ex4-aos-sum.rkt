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
  ;(pretty-display `(indices ,indices))
  (for ([t blockSize])
    (let ([l (for/vector ([i struct-size])
               (vector-ref (vector-ref indices i) t))])
      (unique l)))
  (reg-to-global o O gid)
  )

(define (AOS-sum-test2 threadId blockID blockDim I O I-sizes O-sizes a b c)
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
   (for/bounded
    ((i struct-size))
    (let* ((index
            (modulo
             (+
              (+ (* (@dup i) 0) (quotient (@dup i) a))
              (+ (* localId -1) (quotient localId c))
              -1)
             c))
           (half1 (modulo
              (+
               (+ (* (@dup i) -1) (quotient (@dup i) warpSize))
               (+ (* localId -1) (quotient localId b))
               -1)
              struct-size))
           (lane
            (+
             (modulo
              (+
               (+ (* (@dup i) -1) (quotient (@dup i) warpSize))
               (+ (* localId -1) (quotient localId b))
               -1)
              struct-size)
             (modulo
              (+
               (+ (* (@dup i) -1) (quotient (@dup i) a))
               (+ (* localId a) (quotient localId a))
               warpSize)
              warpSize)))
           (x (shfl (get I-cached index) lane)))
      (pretty-display `(half1 ,i ,half1))
      (pretty-display `(lane ,i ,lane))
      (vector-set! indices i index)
      (accumulate o x #:pred (= localId localId))))
   (reg-to-global o O gid))

(define (AOS-sum-test6-v2 threadId blockID blockDim I O I-sizes O-sizes a b c)
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
   (define lb
     (modulo
      (+ (+ (* localId struct-size) (quotient localId warpSize)) 0)
      warpSize))
   (define ub (modulo (+ lb struct-size) warpSize))
   (define lane (modulo (+ lb (quotient localId b)) warpSize))
   (for/bounded
    ((i struct-size))
    (let* ((index
            (get
             permute
             (modulo
              (+
               (+ (* (@dup i) struct-size) (quotient (@dup i) 1))
               (+ (* localId 0) (quotient localId -1))
               struct-size)
              struct-size)))
           (x (shfl (get I-cached index) lane)))
      (vector-set! indices i index)
      (accumulate o x #:pred (= localId localId))
      (set! lane
        (modulo (+ (+ (* lane warpSize) (quotient lane 1)) 1) warpSize))
      (set! lane (ite (= lane ub) lb lane))))
   (reg-to-global o O gid))

(define (AOS-sum-test6-v1 threadId blockID blockDim I O I-sizes O-sizes a b c)
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
   (define lb (modulo (* localId struct-size) warpSize))
   (define ub (modulo (+ lb struct-size) warpSize))
   (define lane (modulo (+ lb (quotient localId b)) warpSize))
   (for/bounded
    ((i struct-size))
    (let* ((index
            (get
             permute
             (modulo
              (+
               (+ (* (@dup i) c) (quotient (@dup i) -1))
               (+ (* localId 0) (quotient localId -1))
               struct-size)
              struct-size)))
           (x (shfl (get I-cached index) lane)))
      (vector-set! indices i index)
      (accumulate o x #:pred (<= (@dup i) (@dup i)))
      (set! lane
        (modulo (+ (+ (* lane 1) (quotient lane warpSize)) 1) warpSize))
      (set! lane (ite (= lane ub) lb lane))))
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
    (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize))
    (x-y-z (* warpSize struct-size))
    #f)
   (define localId (get-idInWarp threadId))
   (define o (create-accumulator o (list +) identity blockDim))
   (define indices (make-vector struct-size))
   (for/bounded
    ((i struct-size))
    (let* ((index
            (modulo
             (+
              (+ (* (@dup i) warpSize) (quotient (@dup i) b))
              (+ (* localId 0) (quotient localId c))
              struct-size)
             struct-size))
           (lane
            (modulo
             (+
              (+ (* (@dup i) warpSize) (quotient (@dup i) c))
              (+ (* localId struct-size) (quotient localId warpSize))
              0)
             b))
           (x (shfl (get I-cached index) lane)))
      (vector-set! indices i index)
      (accumulate o x #:pred (<= localId localId))))
   (reg-to-global o O gid))


;; fix loop bound, synth load
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

;; m = 3, warpsize = 32: fail when BW=8, need BW>=10

;; fix load
;; m = 3, warpsize = 32: 71/155
;; m = 3, warpsize = 32, constraint col+row permute, distinct?: 33/140
;; m = 3, warpsize = 32, Ras's sketch: 4/10

;; m = 6, warpsize = 32, perm array: 32/1362 s
;; m = 6, warpsize = 32, perm array, strength reduction for lane: 11/388 s (cost 25948)

;; (cost 24924)
;; (cost 23132)
;; (cost 22364)
(define permute (for/vector ([i 64]) #(0 4 1 5 2 3)))
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
                        ;(x-y-z (?warp-offset [(get-x blockID) (get-x blockDim)] [warpID warpSize])) ;; offset
                        (+ (* struct-size blockID blockDim) (* struct-size warpID warpSize))
                        ;(x-y-z (?warp-size warpSize 1)) ;; load size
                        (x-y-z (* warpSize struct-size))
                        #f)

  (define localId (get-idInWarp threadId))
  (define o (create-accumulator o (list +) identity blockDim))
  (define indices (make-vector struct-size))
  ;(define lb (?lane-mod1 localId [a b c struct-size warpSize] 0))
  (define lb (modulo (* localId struct-size) warpSize))
  ;(define ub (?lane-mod2 lb localId [a b c struct-size warpSize] 0))
  (define ub (modulo (+ lb struct-size) warpSize))
  ;(define lane (?lane-mod2 lb localId [a b c struct-size warpSize] 0))
  (define lane (modulo (+ lb (quotient localId b)) warpSize))
  ;(pretty-display `(cost ,(get-cost))) ; (cost 3278)
  (for/bounded ([i struct-size])
    (let* (;[index (modulo (?index localId (@dup i) [a b c struct-size warpSize] 2) struct-size)]
           ;[lane (?lane localId (@dup i) [a b c struct-size warpSize] 4)]
           ;[index (@int (extract (?lane-log (@bv localId) (@dup (@bv i)) [log-a log-b log-c log-m log-n] 2) log-m))]
           ;[lane (@int (?lane-log (@bv localId) (@dup (@bv i)) [log-a log-b log-c log-m log-n] 4))]
           ;[index (?lane-mod2 (@dup i) localId [a b c struct-size warpSize] 0)]
           
           [index (get permute (modulo (- i localId) struct-size))]
           ;[index (get permute (?lane-mod2 (@dup i) localId [a b c struct-size warpSize] 0))]
           ;[lane (+ (modulo (+ i (quotient localId b)) struct-size) (* localId struct-size))]
           ;[lane (?lane-mod2 (@dup i) localId [a b c struct-size warpSize] 1)]
           [x (shfl (get I-cached index) lane)])
      ;(unique-warp (modulo lane warpSize))
      (vector-set! indices i index)
      (accumulate o x #:pred #t) ;(?cond localId (@dup i)))
      ;(set! lane (?lane-mod1 lane [a b c struct-size warpSize] 0))
      ;(set! lane (modulo (+ lane 1) warpSize))
      (set! lane (modulo (+ (+ (* lane (?const a b c struct-size warpSize))
                               (choose 0 (quotient lane (?const a b c struct-size warpSize)))
                               ;0
                               )
                            (?const a b c struct-size warpSize))
                         warpSize))
      (set! lane (ite (= lane ub) lb lane))
      ))
  #;(for ([t blockSize])
    (let ([l (for/list ([i struct-size])
               (vector-ref (vector-ref indices i) t))])
      (unique-list l)))
  
  (reg-to-global o O gid)
  )

; spec 14044
(define (test)
  (for ([w (list 32)])
    (let ([ret (run-with-warp-size AOS-sum-spec AOS-sum-sketch w)])
      (pretty-display `(test ,w ,ret ,(get-cost)))))
  )
;(test)

(define (synthesis)
  (pretty-display "solving...")
  (assert
   (andmap (lambda (w) (run-with-warp-size AOS-sum-spec AOS-sum-sketch w))
           (list 32)))
  (define cost (get-cost))
  ;(assert (= cost 14044))
  (define sol (time (optimize #:minimize (list cost) #:guarantee (assert #t))))

  (define this-cost (evaluate cost sol))
  (print-forms sol)
  (pretty-display `(cost ,this-cost))
  
  ;(define sol2 (solve (assert (< cost this-cost))))
  ;(pretty-display `(cost2 ,(evaluate cost sol2)))
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