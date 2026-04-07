# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from server.env import MedicalTriageEnv
from models import IncidentAction


def test_easy_perfect_score():
    """Agent executes the ideal clinical pathway for a STEMI - should score >=0.80."""
    env = MedicalTriageEnv()
    obs = env.reset(difficulty="easy")

    actions = [
        IncidentAction(action_type="assess",     patient_id="P-101"),
        IncidentAction(action_type="order_test", patient_id="P-101", target="ECG"),
        IncidentAction(action_type="triage",     patient_id="P-101", target="1"),
        IncidentAction(action_type="treat",      patient_id="P-101", target="Aspirin"),
        IncidentAction(action_type="admit",      patient_id="P-101", target="Cardiology"),
    ]

    reward = 0.0
    done = False
    for act in actions:
        obs, reward, done, _ = env.step(act)
        if done:
            break

    for _ in range(15):
        if done:
            break
        obs, reward, done, _ = env.step(IncidentAction(action_type="wait"))

    assert done, "Episode should be done"
    assert reward >= 0.80, "Expected >=0.80 for perfect STEMI pathway, got %s" % reward
    print("[PASS] Easy - perfect STEMI score: %.4f" % reward)


def test_medium_fatal_error_penalizes():
    """Giving Penicillin to a Penicillin-allergic patient must reduce score below 0.50."""
    env = MedicalTriageEnv()
    env.reset(difficulty="medium")

    env.step(IncidentAction(action_type="treat", patient_id="P-102", target="Penicillin"))

    done = False
    reward = 0.0
    for _ in range(25):
        if done:
            break
        _, reward, done, _ = env.step(IncidentAction(action_type="wait"))

    assert len(env.get_state().fatal_errors) > 0, "Fatal error should be recorded"
    assert reward < 0.50, "Fatal interaction should lower score below 0.50, got %.4f" % reward
    print("[PASS] Medium - fatal Penicillin penalty applied: %.4f" % reward)


def test_medium_naloxone_scores_positively():
    """Administering Naloxone to P-108 (Opioid Overdose) should contribute positively."""
    env = MedicalTriageEnv()
    env.reset(difficulty="medium")

    _, r1, _, _ = env.step(IncidentAction(action_type="triage",     patient_id="P-108", target="1"))
    _, r2, _, _ = env.step(IncidentAction(action_type="order_test", patient_id="P-108", target="Tox Screen"))
    _, r3, _, _ = env.step(IncidentAction(action_type="treat",      patient_id="P-108", target="Naloxone"))
    _, r4, _, _ = env.step(IncidentAction(action_type="admit",      patient_id="P-108", target="ICU"))

    step_sum = r1 + r2 + r3 + r4
    assert step_sum > 0, "Step rewards for correct Opioid treatment should be > 0, got %.4f" % step_sum
    print("[PASS] Medium - Naloxone step rewards: %.4f" % step_sum)


def test_state_method():
    """state() should return TriageState with correct fields."""
    env = MedicalTriageEnv()
    env.reset(difficulty="hard")

    s = env.state()
    assert s.episode_id != "", "episode_id should not be empty"
    assert s.step >= 0, "Step should be non-negative, got %d" % s.step
    assert s.max_steps == 25, "Hard task max_steps should be 25, got %d" % s.max_steps
    assert s.done is False, "Should not be done right after reset"
    assert s.difficulty == "hard", "Difficulty should be 'hard', got %s" % s.difficulty
    total = s.patients_in_queue + s.patients_in_beds
    assert total == 3, "Hard task has 3 patients total, got %d" % total
    print("[PASS] state() - episode=%s step=%d patients=%d" % (s.episode_id, s.step, total))


def test_all_tasks_score_in_range():
    """All difficulty levels must produce scores clamped to [0.0, 1.0]."""
    for diff in ("easy", "medium", "hard"):
        env = MedicalTriageEnv()
        env.reset(difficulty=diff)

        done = False
        final_reward = 0.0
        for _ in range(30):
            if done:
                break
            _, final_reward, done, _ = env.step(IncidentAction(action_type="wait"))

        assert 0.0 <= final_reward <= 1.0, \
            "Score out of range for '%s': %.4f" % (diff, final_reward)
        print("[PASS] Score range - %s: %.4f" % (diff, final_reward))


def test_hard_blood_thinner_penalty():
    """Giving Aspirin to a Hemorrhagic Shock patient must register as a fatal error."""
    env = MedicalTriageEnv()
    env.reset(difficulty="hard")

    env.step(IncidentAction(action_type="treat", patient_id="P-104", target="Aspirin"))

    done = False
    reward = 0.0
    for _ in range(30):
        if done:
            break
        _, reward, done, _ = env.step(IncidentAction(action_type="wait"))

    assert len(env.get_state().fatal_errors) > 0, "Aspirin on Hemorrhagic Shock should be a fatal error"
    print("[PASS] Hard - blood thinner penalty registered. Final reward: %.4f" % reward)


if __name__ == "__main__":
    print("=" * 60)
    print("Running Medical Triage Env Test Suite")
    print("=" * 60)

    tests = [
        test_easy_perfect_score,
        test_medium_fatal_error_penalizes,
        test_medium_naloxone_scores_positively,
        test_state_method,
        test_all_tasks_score_in_range,
        test_hard_blood_thinner_penalty,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print("[FAIL] %s: %s" % (t.__name__, e))
            failed += 1
        except Exception as e:
            import traceback
            print("[ERROR] %s: %s" % (t.__name__, e))
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print("Results: %d passed, %d failed" % (passed, failed))
    print("=" * 60)
    if failed:
        sys.exit(1)
