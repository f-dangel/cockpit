"""Schedule Runner to combine DeepOBS and Backboard
using a learning rate schedule."""

import os
import warnings

from torch.optim.lr_scheduler import LambdaLR

from backboard import CockpitPlotter, CockpitTracker
from backpack import backpack, extend
from deepobs.pytorch.runners.runner import PTRunner


class ScheduleCockpitRunner(PTRunner):
    """Schedule Runner to combine DeepOBS and Backboard
    using a learning rate schedule."""

    def __init__(self, optimizer_class, hyperparameter_names):
        super(ScheduleCockpitRunner, self).__init__(
            optimizer_class, hyperparameter_names
        )

    def training(  # noqa: C901
        self,
        tproblem,
        hyperparams,
        num_epochs,
        print_train_iter,
        train_log_interval,
        tb_log,
        tb_log_dir,
        **training_params
    ):

        opt = self._optimizer_class(tproblem.net.parameters(), **hyperparams)

        # Using a LR Scheduler
        lr_sched = training_params["lr_schedule"](num_epochs)
        scheduler = LambdaLR(opt, lr_lambda=lr_sched)

        # Extra Cockpit Stuff #
        # Init Cockpit
        logpath = os.path.join(self._run_directory, self._file_name + "__log")
        cockpit_tracker = CockpitTracker(
            tproblem.net.parameters,
            opt,
            logpath,
            track_interval=training_params["track_interval"],
        )
        cockpit_plotter = CockpitPlotter(logpath)
        # Integrate BackPACK
        extend(tproblem.net)
        tproblem._old_loss = tproblem.loss_function

        def hotfix_lossfunc(reduction="mean"):
            return extend(tproblem._old_loss(reduction=reduction))

        tproblem.loss_function = hotfix_lossfunc
        # End Cockpit Stuff #

        # Lists to log train/test loss and accuracy.
        train_losses = []
        valid_losses = []
        test_losses = []
        train_accuracies = []
        valid_accuracies = []
        test_accuracies = []
        minibatch_train_losses = []
        batch_losses = []  # so it exists the first time we pass it to cockpit

        if tb_log:
            try:
                from torch.utils.tensorboard import SummaryWriter

                summary_writer = SummaryWriter(log_dir=tb_log_dir)
            except ImportError as e:
                warnings.warn(
                    "Not possible to use tensorboard for pytorch. Reason: " + e.msg,
                    RuntimeWarning,
                )
                tb_log = False
        global_step = 0

        for epoch_count in range(num_epochs + 1):
            # Evaluate at beginning of epoch.
            self.evaluate_all(
                epoch_count,
                num_epochs,
                tproblem,
                train_losses,
                valid_losses,
                test_losses,
                train_accuracies,
                valid_accuracies,
                test_accuracies,
            )
            cockpit_tracker.track_epoch(
                epoch_count,
                global_step,
                train_losses[-1],
                valid_losses[-1],
                test_losses[-1],
                train_accuracies[-1],
                valid_accuracies[-1],
                test_accuracies[-1],
            )

            # Break from train loop after the last round of evaluation
            if epoch_count == num_epochs:
                cockpit_tracker.write()
                # Produce the last cockpit view, save it, and optionally show it
                cockpit_plotter.plot(
                    show_plot=training_params["show_plots"], save_plot=True
                )
                break

            # Training #

            # set to training mode
            tproblem.train_init_op()
            batch_count = 0
            while True:
                try:
                    cockpit_tracker.track_before(batch_losses, global_step)

                    batch_losses, _ = tproblem.get_batch_loss_and_accuracy(
                        reduction="mean"  # changed for cockpit
                    )

                    # do zero_grad after forward pass, so we don't set it to
                    # zero when there is no more batch in this epoch
                    opt.zero_grad()

                    # Check if losses is matrix, then sum
                    # This is a hotfix necessary for our current quadratic_deep
                    # implementation.
                    if len(batch_losses.shape) == 2:
                        batch_losses = batch_losses.sum(1)

                    batch_loss = batch_losses.mean()

                    # Use BackPACK for the backward pass, but only use the
                    # extensions necessary.
                    with backpack(*cockpit_tracker.extensions(global_step)):
                        batch_loss.backward(create_graph=True)

                    opt.step()

                    cockpit_tracker.track_after(batch_losses, global_step)

                    if batch_count % train_log_interval == 0:
                        minibatch_train_losses.append(batch_loss.item())
                        if print_train_iter:
                            print(
                                "Epoch {0:d}, step {1:d}: loss {2:g}".format(
                                    epoch_count, batch_count, batch_loss
                                )
                            )
                        if tb_log:
                            summary_writer.add_scalar(
                                "loss", batch_loss.item(), global_step
                            )

                    batch_count += 1
                    global_step += 1

                except StopIteration:
                    break

            # Next step in LR Schedule
            scheduler.step()

            # Write to log file
            cockpit_tracker.write()
            # Create Cockpit Plot if hitting the interval
            if epoch_count % training_params["plot_interval"] == 0:
                cockpit_plotter.plot(
                    show_plot=training_params["show_plots"],
                    save_plot=training_params["save_plots"],
                    savename_append="__epoch__" + str(epoch_count),
                )

        if tb_log:
            summary_writer.close()
        # Put results into output dictionary.
        output = {
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "test_losses": test_losses,
            "minibatch_train_losses": minibatch_train_losses,
            "train_accuracies": train_accuracies,
            "valid_accuracies": valid_accuracies,
            "test_accuracies": test_accuracies,
        }

        return output

    def _add_training_params_to_argparse(self, parser, args, training_params):
        """Add any extra arguments as training parameters.

        Args:
            parser (argparse.ArgumentParser): The argument parser object.
            args (dict): The args that are parsed as locals to the run method.
            training_params (dict): Training parameters that are to read in.
            """
        for tp in training_params:
            args[tp] = training_params[tp]

    def _post_process_output(
        self,
        output,
        testproblem,
        batch_size,
        num_epochs,
        random_seed,
        l2_reg,
        hyperparams,
        **training_params
    ):
        """Remove the training_params from the output.

        Since some training parameters (e.g. the lr_schedule) are not JSON
        serializable, we need to remove them from the output."""

        # remove test accuracy if it is not available
        if "test_accuracies" in output:
            if all(output["test_accuracies"]) == 0:
                del output["test_accuracies"]
                del output["train_accuracies"]
                try:
                    del output["valid_accuracies"]
                except KeyError:
                    pass

        # merge meta data to output dict
        output = {
            "testproblem": testproblem,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "random_seed": random_seed,
            "l2_reg": l2_reg,
            "optimizer_name": self._optimizer_name,
            "optimizer_hyperparams": hyperparams,
            # "training_params": training_params, # removed for Cockpit
            **output,
        }

        return output
