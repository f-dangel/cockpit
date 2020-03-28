"""Custom Runner to track statistics. """

from __future__ import print_function

import time
import warnings

import torch

from backboard import Cockpit
from backpack import backpack, extend
from deepobs.pytorch.runners.runner import PTRunner


class InteractiveCockpitRunner(PTRunner):
    """Custom Runner to track statistics"""

    def __init__(self, optimizer_class, hyperparameter_names):
        super(InteractiveCockpitRunner, self).__init__(
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

        # Lists to log train/test loss and accuracy.
        train_losses = []
        valid_losses = []
        test_losses = []
        train_accuracies = []
        valid_accuracies = []
        test_accuracies = []

        minibatch_train_losses = []

        # Init Cockpit
        cockpit = Cockpit(
            tproblem.net.parameters,
            opt,
            self._run_directory,
            self._file_name,
            track_interval=training_params["track_interval"],
            plot_interval=training_params["plot_interval"],
        )
        batch_losses = []

        if tb_log:
            try:
                from torch.utils.tensorboard import SummaryWriter

                summary_writer = SummaryWriter(log_dir=tb_log_dir)
            except ImportError as err:
                warnings.warn(
                    "Not possible to use tensorboard for pytorch. Reason: " + err.msg,
                    RuntimeWarning,
                )
                tb_log = False
        global_step = 0

        # Integrate BackPACK
        extend(tproblem.net)
        tproblem._old_loss = tproblem.loss_function

        def hotfix_lossfunc(reduction="mean"):
            return extend(tproblem._old_loss(reduction=reduction))

        tproblem.loss_function = hotfix_lossfunc

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
            cockpit.track_epoch(
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
                cockpit.write()
                cockpit.cockpit_plotter.plot(
                    show=training_params["show_plot"], save=True
                )
                break

            # Training #

            # set to training mode
            tproblem.train_init_op()
            batch_count = 0
            while True:
                try:
                    if training_params["track_time"]:
                        if batch_count == 0:
                            comp_time = time.time()
                        elif batch_count % 10 == 0:
                            print("10 iterations took ", time.time() - comp_time)
                            comp_time = time.time()

                    # Cockpit tracking before (only if we hit the track_interval)
                    if cockpit.should_track(global_step):
                        cockpit.track_before(batch_losses, global_step)

                    opt.zero_grad()
                    batch_losses, _ = tproblem.get_batch_loss_and_accuracy(
                        reduction="none"
                    )
                    batch_loss = torch.mean(batch_losses)

                    # Use BackPACK for the backward pass, but only use the
                    # extensions necessary.
                    with backpack(*cockpit.extensions(global_step)):
                        batch_loss.backward(create_graph=True)
                    opt.step()

                    # Cockpit tracking after (only if we hit the track_interval)
                    if cockpit.should_track(global_step):
                        cockpit.track_after(batch_losses, global_step)

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

            # Write to log file if plot_interval or last epoch
            if epoch_count % cockpit.plot_interval == 0:
                # track, but only show if wanted
                cockpit.write()
                cockpit.cockpit_plotter.plot(
                    show=training_params["show_plot"],
                    save=training_params["save_plots"],
                    save_append="__epoch__" + str(epoch_count),
                )

            # Check for any key input during the training,
            # potentially stop training or change optimizers parameters
            stop = cockpit.check_listening()
            if stop:
                break

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
        """Overwrite this method to specify how your
        runner should read in additional training_parameters and to add them to
        argparse.

        Args:
            parser (argparse.ArgumentParser): The argument parser object.
            args (dict): The args that are parsed as locals to the run method.
            training_params (dict): Training parameters that are to read in.
            """
        for tp in training_params:
            args[tp] = training_params[tp]
