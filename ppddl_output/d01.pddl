(define (domain d01)
  (:requirements :strips :typing :negative-preconditions :equality)
  (:types state)
  (:predicates (at ?s - state))

  (:action move-s9-s2-a0
    :parameters ()
    :precondition (at s9)
    :effect (and (not (at s9)) (at s2)))

  (:action move-s2-s9-a0
    :parameters ()
    :precondition (at s2)
    :effect (and (not (at s2)) (at s9)))

  (:action move-s9-s6-a2
    :parameters ()
    :precondition (at s9)
    :effect (and (not (at s9)) (at s6)))

  (:action move-s6-s5-a0
    :parameters ()
    :precondition (at s6)
    :effect (and (not (at s6)) (at s5)))

  (:action move-s9-s1-a3
    :parameters ()
    :precondition (at s9)
    :effect (and (not (at s9)) (at s1)))

  (:action move-s1-s5-a1
    :parameters ()
    :precondition (at s1)
    :effect (and (not (at s1)) (at s5)))

  (:action move-s4-s3-a0
    :parameters ()
    :precondition (at s4)
    :effect (and (not (at s4)) (at s3)))

  (:action move-s3-s5-a0
    :parameters ()
    :precondition (at s3)
    :effect (and (not (at s3)) (at s5)))

  (:action move-s3-s0-a1
    :parameters ()
    :precondition (at s3)
    :effect (and (not (at s3)) (at s0)))

  (:action move-s0-s8-a3
    :parameters ()
    :precondition (at s0)
    :effect (and (not (at s0)) (at s8)))

  (:action move-s4-s1-a1
    :parameters ()
    :precondition (at s4)
    :effect (and (not (at s4)) (at s1)))

  (:action move-s0-s5-a0
    :parameters ()
    :precondition (at s0)
    :effect (and (not (at s0)) (at s5)))

  (:action move-s5-s6-a3
    :parameters ()
    :precondition (at s5)
    :effect (and (not (at s5)) (at s6)))

  (:action move-s3-s4-a1
    :parameters ()
    :precondition (at s3)
    :effect (and (not (at s3)) (at s4)))

  (:action move-s1-s9-a0
    :parameters ()
    :precondition (at s1)
    :effect (and (not (at s1)) (at s9)))

  (:action move-s5-s7-a1
    :parameters ()
    :precondition (at s5)
    :effect (and (not (at s5)) (at s7)))

  (:action move-s8-s0-a3
    :parameters ()
    :precondition (at s8)
    :effect (and (not (at s8)) (at s0)))

  (:action move-s1-s7-a2
    :parameters ()
    :precondition (at s1)
    :effect (and (not (at s1)) (at s7)))

  (:action move-s3-s1-a1
    :parameters ()
    :precondition (at s3)
    :effect (and (not (at s3)) (at s1)))

  (:action move-s4-s9-a1
    :parameters ()
    :precondition (at s4)
    :effect (and (not (at s4)) (at s9)))

  (:action move-s7-s9-a0
    :parameters ()
    :precondition (at s7)
    :effect (and (not (at s7)) (at s9)))

  (:action move-s2-s7-a3
    :parameters ()
    :precondition (at s2)
    :effect (and (not (at s2)) (at s7)))

  (:action move-s3-s6-a3
    :parameters ()
    :precondition (at s3)
    :effect (and (not (at s3)) (at s6)))

  (:action move-s6-s8-a1
    :parameters ()
    :precondition (at s6)
    :effect (and (not (at s6)) (at s8)))

  (:action move-s8-s5-a0
    :parameters ()
    :precondition (at s8)
    :effect (and (not (at s8)) (at s5)))

  (:action move-s5-s8-a1
    :parameters ()
    :precondition (at s5)
    :effect (and (not (at s5)) (at s8)))

  (:action move-s9-s7-a0
    :parameters ()
    :precondition (at s9)
    :effect (and (not (at s9)) (at s7)))

  (:action move-s4-s7-a1
    :parameters ()
    :precondition (at s4)
    :effect (and (not (at s4)) (at s7)))

  (:action move-s0-s4-a0
    :parameters ()
    :precondition (at s0)
    :effect (and (not (at s0)) (at s4)))

  (:action move-s7-s5-a2
    :parameters ()
    :precondition (at s7)
    :effect (and (not (at s7)) (at s5)))

  (:action move-s0-s3-a0
    :parameters ()
    :precondition (at s0)
    :effect (and (not (at s0)) (at s3)))

  (:action move-s8-s6-a1
    :parameters ()
    :precondition (at s8)
    :effect (and (not (at s8)) (at s6)))

  (:action move-s5-s4-a2
    :parameters ()
    :precondition (at s5)
    :effect (and (not (at s5)) (at s4)))

  (:action move-s4-s5-a3
    :parameters ()
    :precondition (at s4)
    :effect (and (not (at s4)) (at s5)))

  (:action move-s5-s0-a0
    :parameters ()
    :precondition (at s5)
    :effect (and (not (at s5)) (at s0)))

  (:action move-s1-s6-a1
    :parameters ()
    :precondition (at s1)
    :effect (and (not (at s1)) (at s6)))

  (:action move-s5-s1-a1
    :parameters ()
    :precondition (at s5)
    :effect (and (not (at s5)) (at s1)))

  (:action move-s1-s3-a1
    :parameters ()
    :precondition (at s1)
    :effect (and (not (at s1)) (at s3)))

  (:action move-s5-s9-a1
    :parameters ()
    :precondition (at s5)
    :effect (and (not (at s5)) (at s9)))

)
