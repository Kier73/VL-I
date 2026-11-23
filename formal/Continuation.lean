import Init

namespace Continuation

universe u v r o
variable {S : Type u} {A : Type v} {R : Type r} {O : Type o}

/-- F(s, []) = s and F(s, a :: w) = F(step(s,a), w). -/
def run (step : S → A → S) (s : S) (w : List A) : S :=
  w.foldl step s

/-- π ∘ step_a = advance_a ∘ π implies π ∘ F_w = G_w ∘ π for every word. -/
theorem liftRun (step : S → A → S) (advance : R → A → R) (project : S → R)
    (transition : ∀ s a, project (step s a) = advance (project s) a)
    (s : S) (w : List A) :
    project (run step s w) = run advance (project s) w := by
  induction w generalizing s with
  | nil => rfl
  | cons a w ih =>
    change project (run step (step s a) w) = run advance (advance (project s) a) w
    rw [ih, transition]

/-- Equality in a closed representation implies equality of every future readout. -/
theorem futureEqual (step : S → A → S) (advance : R → A → R)
    (project : S → R) (read : S → O) (decode : R → O)
    (transition : ∀ s a, project (step s a) = advance (project s) a)
    (observable : ∀ s, read s = decode (project s))
    (s t : S) (same : project s = project t) (w : List A) :
    read (run step s w) = read (run step t w) := by
  exact (observable _).trans ((congrArg decode (liftRun step advance project transition s w)).trans
    ((congrArg (fun r => decode (run advance r w)) same).trans
      ((congrArg decode (liftRun step advance project transition t w).symm).trans (observable _).symm)))

end Continuation
