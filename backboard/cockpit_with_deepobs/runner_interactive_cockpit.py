"""Custom Runner to track statistics. """

from __future__ import print_function

import warnings

import torch

from backboard import Cockpit
from backpack import backpack, extend, extensions
from deepobs.pytorch.runners.runner import PTRunner


class CockpitRunner(PTRunner):
    """Custom Runner to track statistics"""

    def __init__(self, optimizer_class, hyperparameter_names):
        super(CockpitRunner, self).__init__(optimizer_class, hyperparameter_names)

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
            self._run_directory,
            self._file_name,
            plot_interval=training_params["plot_interval"],
        )

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
            cockpit.epoch_track(train_accuracies, valid_accuracies, opt)

            # Break from train loop after the last round of evaluation
            if epoch_count == num_epochs:
                cockpit.write()
                # always draw the last one, but only show if necessary
                cockpit.cockpit_plotter.plot(draw=training_params["show_plot"])
                break

            # Training #

            # set to training mode
            tproblem.train_init_op()
            batch_count = 0
            while True:
                try:
                    opt.zero_grad()
                    batch_losses, _ = tproblem.get_batch_loss_and_accuracy(
                        reduction="none"
                    )
                    batch_loss = torch.mean(batch_losses)
                    if batch_count % train_log_interval == 0:
                        bp_extensions = (
                            extensions.Variance(),
                            extensions.BatchGrad(),
                            extensions.DiagHessian(),
                        )
                    else:
                        bp_extensions = []
                        # eigenvalue calc
                    with backpack(*bp_extensions):
                        batch_loss.backward()
                        if batch_count % train_log_interval == 0:
                            cockpit.track(batch_losses)
                    opt.step()

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
                if training_params["show_plot"]:
                    cockpit.cockpit_plotter.plot()

            # Check for any key input during the training,
            # potentially stop training or change optimizers parameters
            stop = cockpit.check_listening(opt)
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

        cockpit.cockpit_plotter.save_plot()
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
