"""Base class for executing and hooking into a training loop to execute checks."""


from backpack import extend

from cockpit import Cockpit
from tests.utils.rand import restore_rng_state


class SimpleTestHarness:
    """Class for running a simple test loop with the Cockpit.

    Args:
        problem (string): The (instantiated) problem to test on.
    """

    def __init__(self, problem):
        """Store the instantiated problem."""
        self.problem = problem

    def test(self, cockpit_kwargs, *backpack_exts):
        """Run the test loop.

        Args:
            cockpit_kwargs (dict): Arguments for the cockpit.
            *backpack_exts (list): List of user-defined BackPACK extensions.
        """
        problem = self.problem

        data = problem.data
        device = problem.device
        iterations = problem.iterations

        # Extend
        model = extend(problem.model)
        loss_fn = extend(problem.loss_function)
        individual_loss_fn = extend(problem.individual_loss_function)

        # Create Optimizer
        optimizer = problem.optimizer

        # Initialize Cockpit
        self.cockpit = Cockpit(model.parameters(), **cockpit_kwargs)

        # print(cockpit_exts)

        # Main training loop
        global_step = 0
        for inputs, labels in iter(data):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            losses = individual_loss_fn(outputs, labels)

            # code inside this block does not alter random number generation
            with restore_rng_state():
                # backward pass
                with self.cockpit(
                    global_step,
                    *backpack_exts,
                    info={
                        "batch_size": inputs.shape[0],
                        "individual_losses": losses,
                        "loss": loss,
                        "optimizer": optimizer,
                    },
                ):
                    loss.backward(create_graph=self.cockpit.create_graph(global_step))
                    self.check_in_context()

                self.check_after_context()

            # optimizer step
            optimizer.step()
            global_step += 1

            if global_step >= iterations:
                break

    def check_in_context(self):
        """Check that will be executed within the cockpit context."""
        pass

    def check_after_context(self):
        """Check that will be executed directly after the cockpit context."""
        pass
