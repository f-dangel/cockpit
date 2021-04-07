"""Schedule Runner using a learning rate schedule."""

import os
import warnings

from backobs import extend_with_access_unreduced_loss
from deepobs.pytorch.runners.runner import PTRunner
from torch.optim.lr_scheduler import LambdaLR

from cockpit import Cockpit
from cockpit.plotter import CockpitPlotter
from cockpit.utils import schedules


class _DeepOBSRunner(PTRunner):
    """Abstract schedule Runner using a learning rate schedule.

    This class serves as base class for running regular DeepOBS experiments with the
    cockpit, as well as memory/runtime benchmarks and tests.
    """

    # TODO Move arguments 'quantities', 'plot', 'plot_schedule', 'secondary_screen'
    # to self.training
    def __init__(
        self,
        optimizer_class,
        hyperparameter_names,
        quantities=None,
        plot=True,
        plot_schedule=None,
        secondary_screen=False,
    ):
        """Initialize the runner.

        Args:
            optimizer_class (torch.optim): The optimizer.
            hyperparameter_names (dict): Hyperparameters of the optimizer.
            quantities ([Quantity], optional): List of quantities used by the
                cockpit. Use all quantities by default.
            plot (bool, optional): Whether results should be plotted.
            plot_schedule (callable): Function that maps an iteration to a boolean
                which determines if a plot should be created and tracked data output
                should be written.
            secondary_screen (bool): Whether to plot other experimental quantities
                on a secondary screen.
        """
        super().__init__(optimizer_class, hyperparameter_names)
        self._quantities = quantities
        self._enable_plotting = plot
        self._plot_schedule = plot_schedule
        self._secondary_screen = secondary_screen

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

        # Integrate BackPACK
        extend_with_access_unreduced_loss(tproblem)

        trainable_params = [p for p in tproblem.net.parameters() if p.requires_grad]
        cockpit = Cockpit(trainable_params, quantities=self._quantities)

        plotter = CockpitPlotter(secondary_screen=self._secondary_screen)
        if self._plot_schedule is not None:
            plot_schedule = self._plot_schedule
        else:
            warnings.warn(
                "You are using plot_interval, which will be deprecated. "
                + "Use plot_schedule instead"
            )
            plot_schedule = schedules.linear(training_params["plot_interval"])

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
            if self._should_eval():
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
                break

            # Training #

            # set to training mode
            tproblem.train_init_op()
            batch_count = 0
            while True:
                try:
                    opt.zero_grad()

                    batch_loss, _ = tproblem.get_batch_loss_and_accuracy(
                        reduction="mean"
                    )

                    info = {
                        "batch_size": self._extract_batch_size(batch_loss),
                        "individual_losses": self._extract_individual_losses(
                            batch_loss,
                        ),
                        "loss": batch_loss,
                        "optimizer": opt,
                    }

                    # COCKPIT: Use necessary BackPACK extensions and track #
                    with cockpit(global_step, info=info):
                        batch_loss.backward(
                            create_graph=cockpit.create_graph(global_step)
                        )

                    if plot_schedule(global_step):
                        plotter.plot(
                            cockpit,
                            savedir=logpath,
                            show_plot=training_params["show_plots"],
                            save_plot=training_params["save_plots"],
                            savename_append="__epoch__"
                            + str(epoch_count).zfill(len(str(num_epochs)))
                            + "__global_step__"
                            + str(global_step).zfill(6),
                        )

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

                    self._maybe_stop_iteration(global_step, batch_count)

                except StopIteration:
                    break

            # Next step in LR Schedule
            scheduler.step()

        # COCKPIT: Write to file and optionally plot after last epoch #
        cockpit.write(logpath)

        if self._enable_plotting:
            plotter.plot(
                cockpit,
                savedir=logpath,
                show_plot=training_params["show_plots"],
                save_plot=training_params["save_final_plot"],
            )

            if training_params["save_animation"]:
                plotter.build_animation(logpath)

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

    @staticmethod
    def _extract_batch_size(batch_loss):
        """Return the batch size from individual losses in ``batch_loss``.

        Note:
            Requires access to unreduced losses made accessible by BackOBS.

        Args:
            batch_loss (torch.Tensor): Mini-batch loss computed from a forward
                of a DeepOBS testproblem extended by BackOBS.

        Returns:
            int: Mini-batch size
        """
        return len(batch_loss._unreduced_loss)

    @staticmethod
    def _extract_individual_losses(batch_loss):
        """A simple hotfix for the unreduced loss values.

        For the quadratic_deep problem of DeepOBS, the unreduced losses are a matrix
        and should be averaged over the second axis.

        Args:
            batch_loss (torch.Tensor): Mini-batch loss from current step, with the
                unreduced losses as an attribute.

        Returns:
            torch.Tensor: (Averaged) individual losses.
        """
        batch_losses = batch_loss._unreduced_loss

        if len(batch_losses.shape) == 2:
            batch_losses = batch_losses.mean(1)

        return batch_losses

    def _maybe_stop_iteration(self, global_step, batch_count):
        """Optionally stop iteration of an epoch earlier.

        The iteration will be stopped if a ``StopIteration`` exception is raised.

        Args:
            global_step (int): Number of iterations performed in total.
            batch_count (int): Number of mini-batches drawn in the current epoch.

        Raises:
            NotImplementedError: If not implemented. Should be overwritten by subclass.
        """
        raise NotImplementedError()

    def _should_eval(self):
        """Return bool that determines computation of DeepOBS' metrics on large sets.

        This method can be used to disable the computation of train/test/valid losses
        and accuracies at the beginning of each epoch. The computation will be switched
        off if this function returns ``False``.

        Raises:
            NotImplementedError: If not implemented. Should be overwritten by subclass.
        """
        raise NotImplementedError()


class DeepOBSRunner(_DeepOBSRunner):
    """Schedule Runner using a learning rate schedule."""

    def _maybe_stop_iteration(self, global_step, batch_count):
        """Never stop the default DeepOBS iteration of an epoch."""
        pass

    def _should_eval(self):
        """Enable DeepOBS' evaluation of test/train/valid losses and accuracies."""
        return True
