(define (domain d01)
  (:requirements :strips :typing :negative-preconditions :equality)
  (:types state)
  (:predicates (at ?s - state))

  (:action move-s14-s20-a0
    :parameters ()
    :precondition (at s14)
    :effect (and (not (at s14)) (at s20)))

  (:action move-s20-s28-a1
    :parameters ()
    :precondition (at s20)
    :effect (and (not (at s20)) (at s28)))

  (:action move-s28-s20-a3
    :parameters ()
    :precondition (at s28)
    :effect (and (not (at s28)) (at s20)))

  (:action move-s20-s16-a3
    :parameters ()
    :precondition (at s20)
    :effect (and (not (at s20)) (at s16)))

  (:action move-s21-s3-a0
    :parameters ()
    :precondition (at s21)
    :effect (and (not (at s21)) (at s3)))

  (:action move-s3-s29-a3
    :parameters ()
    :precondition (at s3)
    :effect (and (not (at s3)) (at s29)))

  (:action move-s29-s25-a0
    :parameters ()
    :precondition (at s29)
    :effect (and (not (at s29)) (at s25)))

  (:action move-s25-s6-a2
    :parameters ()
    :precondition (at s25)
    :effect (and (not (at s25)) (at s6)))

  (:action move-s17-s14-a0
    :parameters ()
    :precondition (at s17)
    :effect (and (not (at s17)) (at s14)))

  (:action move-s20-s6-a2
    :parameters ()
    :precondition (at s20)
    :effect (and (not (at s20)) (at s6)))

  (:action move-s20-s9-a3
    :parameters ()
    :precondition (at s20)
    :effect (and (not (at s20)) (at s9)))

  (:action move-s9-s28-a0
    :parameters ()
    :precondition (at s9)
    :effect (and (not (at s9)) (at s28)))

  (:action move-s28-s24-a0
    :parameters ()
    :precondition (at s28)
    :effect (and (not (at s28)) (at s24)))

  (:action move-s2-s14-a3
    :parameters ()
    :precondition (at s2)
    :effect (and (not (at s2)) (at s14)))

  (:action move-s14-s17-a3
    :parameters ()
    :precondition (at s14)
    :effect (and (not (at s14)) (at s17)))

  (:action move-s17-s0-a1
    :parameters ()
    :precondition (at s17)
    :effect (and (not (at s17)) (at s0)))

  (:action move-s0-s9-a0
    :parameters ()
    :precondition (at s0)
    :effect (and (not (at s0)) (at s9)))

  (:action move-s9-s11-a1
    :parameters ()
    :precondition (at s9)
    :effect (and (not (at s9)) (at s11)))

  (:action move-s11-s28-a0
    :parameters ()
    :precondition (at s11)
    :effect (and (not (at s11)) (at s28)))

  (:action move-s28-s13-a0
    :parameters ()
    :precondition (at s28)
    :effect (and (not (at s28)) (at s13)))

  (:action move-s13-s18-a1
    :parameters ()
    :precondition (at s13)
    :effect (and (not (at s13)) (at s18)))

  (:action move-s2-s21-a3
    :parameters ()
    :precondition (at s2)
    :effect (and (not (at s2)) (at s21)))

  (:action move-s21-s14-a2
    :parameters ()
    :precondition (at s21)
    :effect (and (not (at s21)) (at s14)))

  (:action move-s14-s3-a1
    :parameters ()
    :precondition (at s14)
    :effect (and (not (at s14)) (at s3)))

  (:action move-s29-s28-a1
    :parameters ()
    :precondition (at s29)
    :effect (and (not (at s29)) (at s28)))

  (:action move-s28-s25-a3
    :parameters ()
    :precondition (at s28)
    :effect (and (not (at s28)) (at s25)))

  (:action move-s25-s7-a1
    :parameters ()
    :precondition (at s25)
    :effect (and (not (at s25)) (at s7)))

  (:action move-s20-s25-a1
    :parameters ()
    :precondition (at s20)
    :effect (and (not (at s20)) (at s25)))

  (:action move-s21-s2-a1
    :parameters ()
    :precondition (at s21)
    :effect (and (not (at s21)) (at s2)))

  (:action move-s14-s21-a2
    :parameters ()
    :precondition (at s14)
    :effect (and (not (at s14)) (at s21)))

  (:action move-s29-s22-a2
    :parameters ()
    :precondition (at s29)
    :effect (and (not (at s29)) (at s22)))

  (:action move-s22-s25-a2
    :parameters ()
    :precondition (at s22)
    :effect (and (not (at s22)) (at s25)))

  (:action move-s5-s17-a2
    :parameters ()
    :precondition (at s5)
    :effect (and (not (at s5)) (at s17)))

  (:action move-s0-s11-a0
    :parameters ()
    :precondition (at s0)
    :effect (and (not (at s0)) (at s11)))

  (:action move-s28-s18-a0
    :parameters ()
    :precondition (at s28)
    :effect (and (not (at s28)) (at s18)))

  (:action move-s14-s9-a3
    :parameters ()
    :precondition (at s14)
    :effect (and (not (at s14)) (at s9)))

  (:action move-s17-s5-a0
    :parameters ()
    :precondition (at s17)
    :effect (and (not (at s17)) (at s5)))

  (:action move-s0-s17-a0
    :parameters ()
    :precondition (at s0)
    :effect (and (not (at s0)) (at s17)))

  (:action move-s0-s20-a3
    :parameters ()
    :precondition (at s0)
    :effect (and (not (at s0)) (at s20)))

  (:action move-s20-s11-a0
    :parameters ()
    :precondition (at s20)
    :effect (and (not (at s20)) (at s11)))

  (:action move-s28-s7-a0
    :parameters ()
    :precondition (at s28)
    :effect (and (not (at s28)) (at s7)))

  (:action move-s3-s22-a0
    :parameters ()
    :precondition (at s3)
    :effect (and (not (at s3)) (at s22)))

  (:action move-s25-s22-a0
    :parameters ()
    :precondition (at s25)
    :effect (and (not (at s25)) (at s22)))

  (:action move-s25-s29-a2
    :parameters ()
    :precondition (at s25)
    :effect (and (not (at s25)) (at s29)))

  (:action move-s29-s6-a1
    :parameters ()
    :precondition (at s29)
    :effect (and (not (at s29)) (at s6)))

  (:action move-s5-s2-a2
    :parameters ()
    :precondition (at s5)
    :effect (and (not (at s5)) (at s2)))

  (:action move-s25-s4-a2
    :parameters ()
    :precondition (at s25)
    :effect (and (not (at s25)) (at s4)))

  (:action move-s4-s26-a0
    :parameters ()
    :precondition (at s4)
    :effect (and (not (at s4)) (at s26)))

  (:action move-s11-s8-a3
    :parameters ()
    :precondition (at s11)
    :effect (and (not (at s11)) (at s8)))

  (:action move-s8-s16-a0
    :parameters ()
    :precondition (at s8)
    :effect (and (not (at s8)) (at s16)))

  (:action move-s16-s18-a0
    :parameters ()
    :precondition (at s16)
    :effect (and (not (at s16)) (at s18)))

  (:action move-s18-s16-a2
    :parameters ()
    :precondition (at s18)
    :effect (and (not (at s18)) (at s16)))

  (:action move-s16-s7-a1
    :parameters ()
    :precondition (at s16)
    :effect (and (not (at s16)) (at s7)))

  (:action move-s7-s16-a2
    :parameters ()
    :precondition (at s7)
    :effect (and (not (at s7)) (at s16)))

  (:action move-s7-s18-a2
    :parameters ()
    :precondition (at s7)
    :effect (and (not (at s7)) (at s18)))

  (:action move-s18-s7-a1
    :parameters ()
    :precondition (at s18)
    :effect (and (not (at s18)) (at s7)))

  (:action move-s18-s8-a2
    :parameters ()
    :precondition (at s18)
    :effect (and (not (at s18)) (at s8)))

  (:action move-s8-s18-a1
    :parameters ()
    :precondition (at s8)
    :effect (and (not (at s8)) (at s18)))

  (:action move-s19-s21-a1
    :parameters ()
    :precondition (at s19)
    :effect (and (not (at s19)) (at s21)))

  (:action move-s29-s9-a3
    :parameters ()
    :precondition (at s29)
    :effect (and (not (at s29)) (at s9)))

  (:action move-s17-s15-a0
    :parameters ()
    :precondition (at s17)
    :effect (and (not (at s17)) (at s15)))

  (:action move-s15-s10-a1
    :parameters ()
    :precondition (at s15)
    :effect (and (not (at s15)) (at s10)))

  (:action move-s29-s7-a0
    :parameters ()
    :precondition (at s29)
    :effect (and (not (at s29)) (at s7)))

  (:action move-s5-s15-a0
    :parameters ()
    :precondition (at s5)
    :effect (and (not (at s5)) (at s15)))

  (:action move-s25-s26-a3
    :parameters ()
    :precondition (at s25)
    :effect (and (not (at s25)) (at s26)))

  (:action move-s26-s1-a0
    :parameters ()
    :precondition (at s26)
    :effect (and (not (at s26)) (at s1)))

  (:action move-s21-s12-a0
    :parameters ()
    :precondition (at s21)
    :effect (and (not (at s21)) (at s12)))

  (:action move-s12-s3-a0
    :parameters ()
    :precondition (at s12)
    :effect (and (not (at s12)) (at s3)))

  (:action move-s22-s29-a1
    :parameters ()
    :precondition (at s22)
    :effect (and (not (at s22)) (at s29)))

  (:action move-s8-s13-a3
    :parameters ()
    :precondition (at s8)
    :effect (and (not (at s8)) (at s13)))

  (:action move-s13-s27-a3
    :parameters ()
    :precondition (at s13)
    :effect (and (not (at s13)) (at s27)))

  (:action move-s27-s16-a0
    :parameters ()
    :precondition (at s27)
    :effect (and (not (at s27)) (at s16)))

  (:action move-s9-s8-a0
    :parameters ()
    :precondition (at s9)
    :effect (and (not (at s9)) (at s8)))

  (:action move-s8-s23-a1
    :parameters ()
    :precondition (at s8)
    :effect (and (not (at s8)) (at s23)))

  (:action move-s19-s12-a0
    :parameters ()
    :precondition (at s19)
    :effect (and (not (at s19)) (at s12)))

  (:action move-s12-s22-a1
    :parameters ()
    :precondition (at s12)
    :effect (and (not (at s12)) (at s22)))

  (:action move-s22-s1-a2
    :parameters ()
    :precondition (at s22)
    :effect (and (not (at s22)) (at s1)))

  (:action move-s1-s4-a0
    :parameters ()
    :precondition (at s1)
    :effect (and (not (at s1)) (at s4)))

  (:action move-s29-s24-a2
    :parameters ()
    :precondition (at s29)
    :effect (and (not (at s29)) (at s24)))

  (:action move-s9-s0-a3
    :parameters ()
    :precondition (at s9)
    :effect (and (not (at s9)) (at s0)))

  (:action move-s9-s10-a3
    :parameters ()
    :precondition (at s9)
    :effect (and (not (at s9)) (at s10)))

  (:action move-s10-s28-a3
    :parameters ()
    :precondition (at s10)
    :effect (and (not (at s10)) (at s28)))

  (:action move-s13-s8-a0
    :parameters ()
    :precondition (at s13)
    :effect (and (not (at s13)) (at s8)))

  (:action move-s14-s0-a3
    :parameters ()
    :precondition (at s14)
    :effect (and (not (at s14)) (at s0)))

  (:action move-s15-s0-a0
    :parameters ()
    :precondition (at s15)
    :effect (and (not (at s15)) (at s0)))

  (:action move-s0-s10-a3
    :parameters ()
    :precondition (at s0)
    :effect (and (not (at s0)) (at s10)))

  (:action move-s10-s13-a1
    :parameters ()
    :precondition (at s10)
    :effect (and (not (at s10)) (at s13)))

  (:action move-s11-s10-a0
    :parameters ()
    :precondition (at s11)
    :effect (and (not (at s11)) (at s10)))

  (:action move-s22-s24-a0
    :parameters ()
    :precondition (at s22)
    :effect (and (not (at s22)) (at s24)))

  (:action move-s11-s20-a0
    :parameters ()
    :precondition (at s11)
    :effect (and (not (at s11)) (at s20)))

  (:action move-s2-s19-a2
    :parameters ()
    :precondition (at s2)
    :effect (and (not (at s2)) (at s19)))

  (:action move-s19-s2-a1
    :parameters ()
    :precondition (at s19)
    :effect (and (not (at s19)) (at s2)))

  (:action move-s1-s25-a3
    :parameters ()
    :precondition (at s1)
    :effect (and (not (at s1)) (at s25)))

  (:action move-s2-s5-a3
    :parameters ()
    :precondition (at s2)
    :effect (and (not (at s2)) (at s5)))

  (:action move-s5-s14-a3
    :parameters ()
    :precondition (at s5)
    :effect (and (not (at s5)) (at s14)))

  (:action move-s11-s6-a2
    :parameters ()
    :precondition (at s11)
    :effect (and (not (at s11)) (at s6)))

  (:action move-s10-s11-a3
    :parameters ()
    :precondition (at s10)
    :effect (and (not (at s10)) (at s11)))

  (:action move-s4-s25-a0
    :parameters ()
    :precondition (at s4)
    :effect (and (not (at s4)) (at s25)))

  (:action move-s21-s19-a1
    :parameters ()
    :precondition (at s21)
    :effect (and (not (at s21)) (at s19)))

  (:action move-s3-s12-a2
    :parameters ()
    :precondition (at s3)
    :effect (and (not (at s3)) (at s12)))

  (:action move-s28-s8-a0
    :parameters ()
    :precondition (at s28)
    :effect (and (not (at s28)) (at s8)))

  (:action move-s12-s1-a0
    :parameters ()
    :precondition (at s12)
    :effect (and (not (at s12)) (at s1)))

  (:action move-s25-s1-a2
    :parameters ()
    :precondition (at s25)
    :effect (and (not (at s25)) (at s1)))

)
