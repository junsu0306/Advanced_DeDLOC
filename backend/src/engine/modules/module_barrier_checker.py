class ModuleBarrierChecker:
    def __init__(self):
        self.name = "BarrierChecker"

    def run(self, engine, task):
        if getattr(task, "test_field_", 0) >= 60:
            print("Found no problem, terminating...")
            engine.schedule_after_use(engine.module_d2h_copy_, task)
        else:
            idx_to_check = 1 if engine.local_rank() == 0 else 0
            if task.test_field_ % 2 == 0:
                task.shared_props_.shared_test_field_[engine.local_rank()] = task.test_field_
            else:
                target = task.shared_props_.shared_test_field_[idx_to_check]
                assert target == task.test_field_ - 1
            task.test_field_ += 1

            engine.schedule_after_barrier(engine.module_barrier_checker_, task, self.name + str(task.test_field_))

        return "RESULT_SUCCESS"
