"""Schedule Runner using a learning rate schedule."""

import os
import warnings

from torch.optim.lr_scheduler import LambdaLR

from backboard import Cockpit
from deepobs.pytorch.runners.runner import PTRunner


class ScheduleCockpitRunner(PTRunner):
    """Schedule Runner using a learning rate schedule."""

    def __init__(self, optimizer_class, hyperparameter_names, quantities=None):
        """Initialize the runner.

        Args:
            optimizer_class (torch.optim): The optimizer.
            hyperparameter_names (dict): Hyperparameters of the optimizer.
            quantities ([Quantity], optional): List of quantities used by the
                cockpit. Use all quantities by default.
        """
        super(ScheduleCockpitRunner, self).__init__(
            optimizer_class, hyperparameter_names
        )
        self._quantities = quantities

    def training(  # noqa: C901
        self,
        tproblem,
        hyperparams,
        num_epochs,
        print_train_iter,
        train_log_interval,
        tb_log,
        tb_log_dir,
        **training_params,
    ):
        """Training loop for this runner.

        Args:
            tproblem (deepobs.pytorch.testproblems.testproblem): The testproblem
                instance to train on.
            hyperparams (dict): The optimizer hyperparameters to use for the training.
            num_epochs (int): The number of training epochs.
            print_train_iter (bool): Whether to print the training progress at
                every train_log_interval
            train_log_interval (int): Mini-batch interval for logging.
            tb_log (bool): Whether to use tensorboard logging or not
            tb_log_dir (str): The path where to save tensorboard events.
            **training_params (dict): Kwargs for additional training parameters
                that will be used by the cockpit.

        Returns:
            dict: Output of the training loop
        """
        opt = self._optimizer_class(tproblem.net.parameters(), **hyperparams)

        # Using a LR Scheduler
        lr_sched = training_params["lr_schedule"](num_epochs)
        scheduler = LambdaLR(opt, lr_lambda=lr_sched)

        # COCKPIT: Initialize it #
        logpath = self._get_cockpit_logpath()
        cockpit = Cockpit(
            tproblem,
            logpath,
            training_params["track_interval"],
            quantities=self._quantities,
        )

        # Lists to log train/test loss and accuracy.
        train_losses = []
        valid_losses = []
        test_losses = []
        train_accuracies = []
        valid_accuracies = []
        test_accuracies = []
        minibatch_train_losses = []

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

            # COCKPIT: Log already computed quantities #
            cockpit.log(
                global_step,
                epoch_count,
                train_losses[-1],
                valid_losses[-1],
                test_losses[-1],
                train_accuracies[-1],
                valid_accuracies[-1],
                test_accuracies[-1],
                opt.param_groups[0]["lr"],
            )

            # Break from train loop after the last round of evaluation
            if epoch_count == num_epochs:
                # COCKPIT: Write to file and optionally plot after last epoch #
                cockpit.write()
                cockpit.plot(
                    training_params["show_plots"],
                    training_params["save_final_plot"],
                )
                if training_params["save_animation"]:
                    cockpit.build_animation()
                break

            # Training #

            # set to training mode
            tproblem.train_init_op()
            batch_count = 0
            while True:
                try:
                    opt.zero_grad()

                    # COCKPIT: Use necessary BackPACK extensions and track #
                    with cockpit(global_step):
                        batch_loss, _ = tproblem.get_batch_loss_and_accuracy(
                            reduction="mean"
                        )
                        batch_loss.backward(create_graph=cockpit.create_graph)

                    cockpit.track(global_step, batch_loss)

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

            # Next step in LR Schedule
            scheduler.step()

            # COCKPIT: Write to file and optionally plot after each epoch #
            cockpit.write()
            if epoch_count % training_params["plot_interval"] == 0:
                cockpit.plot(
                    training_params["show_plots"],
                    training_params["save_plots"],
                    savename_append="__epoch__"
                    + str(epoch_count).zfill(len(str(num_epochs))),
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
        **training_params,
    ):
        """Remove the training_params from the output.

        Since some training parameters (e.g. the lr_schedule) are not JSON
        serializable, we need to remove them from the output.

        Args:
            output ([type]): [description]
            testproblem (deepobs.pytorch.testproblems.testproblem): The testproblem
                instance to train on.
            batch_size (int): Batch size of the problem.
            num_epochs (int): The number of training epochs.
            random_seed (int): Random seed of this run.
            l2_reg (float): L2 regularization of the problem.
            hyperparams (dict): Hyperparameters of the optimizer.
            **training_params (dict): Kwargs of the training parameters.

        Returns:
            dict: Output of the training loop.
        """
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

    def _get_cockpit_logpath(self):
        """Return logpath for cockpit."""
        return os.path.join(self._run_directory, self._file_name + "__log")
