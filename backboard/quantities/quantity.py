"""Base class for a tracked quantity."""

from collections import defaultdict

import torch


class Quantity:
    """Base class for a tracked quantity with the Cockpit.

    Quantities can modify the backward pass:
    1. They can ask that the forward pass computation graph be restored.
       This may be useful if their computation requires differentiating through
       the mini-batch loss.
    2. They can ask for certain BackPACK extensions being computed.
    """

    def __init__(self, track_interval=1, verbose=False):
        """Initialize the Quantity by storing the track interval.

        Crucially, it creates the output dictionary, that is meant to store all
        values that should be stored.

        Args:
            track_interval (int, optional): Tracking rate. Defaults to 1.
            verbose (bool, optional): Turns on verbose mode. Defaults to ``False``.
        """
        self._track_interval = track_interval
        self._verbose = verbose
        self.output = defaultdict(dict)

    def create_graph(self, global_step):
        """Return whether access to the forward pass computation graph is needed.

        Args:
            global_step (int): The current iteration number.

        Returns:
            bool: ``True`` if the computation graph shall not be deleted,
                else ``False``.
        """
        return False

    def extensions(self, global_step):
        """Return list of BackPACK extensions required for the computation.

        Args:
            global_step (int): The current iteration number.

        Returns:
            list: (Potentially empty) list with required BackPACK quantities.
        """
        raise NotImplementedError

    def compute(self, global_step, params, batch_loss):
        """Evaluate quantity at a step in training.

        Args:
            global_step (int): The current iteration number.
            params ([torch.Tensor]): List of torch.Tensors holding the network's
                parameters.
            batch_loss (torch.Tensor): Mini-batch loss from current step.
        """
        raise NotImplementedError

    @staticmethod
    def _fetch_grad(params, aggregate=False):
        """Return parameter gradients.

        Gradients can be aggregated among parameters (vectorized). In this case,
        the return value is a vector of shape ``[D]`` where ``D`` is the total
        number of parameters.

        Args:
            params ([torch.Tensor]): List of parameters whose gradient will be fetched.
            aggregate (bool): Flatten and concatenate all gradients.

        Returns:
            [torch.Tensor] if ``aggregate`` is False: List of same length as ``params``
                with items of same shape containing the gradients.
            torch.Tensor if ``aggregate`` is True: Gradient of vectorized parameters.
        """
        grads = [p.grad for p in params]

        if aggregate:
            grads = torch.cat([g.flatten() for g in grads])

        return grads

    @staticmethod
    def _fetch_grad_l2_squared(params, aggregate=False):
        """Return gradient squared ℓ₂ norm, || ∇L ||₂².

        Norms can be aggregated among parameters (summed). In this case, the return
        value is a scalar. Without aggregation, the parameter-wise squared gradient
        ℓ₂ norm is returned

        Args:
            params ([torch.Tensor]): List of parameters whose gradient squared ℓ₂ norm
                will be fetched.
            aggregate (bool): Sum squared norms over parameter blocks.

        Returns:
            [torch.Tensor] if ``aggregate`` is False: List of same length as ``params``
                with scalar items containing block-wise gradient squared norms.
            torch.Tensor if ``aggregate`` is True: A scalar, the squared gradient norm.
        """
        grad_l2_squared = [p.grad.pow(2).sum() for p in params]

        if aggregate:
            grad_l2_squared = sum(grad_l2_squared)

        return grad_l2_squared

    @staticmethod
    def _fetch_batch_grad(params, aggregate=False):
        """Return individual gradients from ``backpack.extensions.BatchGrad``.

        Individual gradients can be aggregated among parameters. With a batch size
        ``N``, the aggregated individual gradients of a ``D``-dimensional model have
        shape ``[N, D]``.

        Args:
            params ([torch.Tensor]): List of parameters whose gradient squared ℓ₂ norm
                will be fetched.
            aggregate (bool): Flatten and concatenate individual gradients over the
                parameter dimension.

        Returns:
            [torch.Tensor] if ``aggregate`` is False: List of same length as ``params``
                with items of shape ``[N, *]`` containing block-wise individual
                gradients. Here, ``*`` denotes the shape of the associated parameter
                block.
            torch.Tensor if ``aggregate`` is True: A vector of shape ``[N, D]`` with
                individual gradients for the vectorized model parameters.
        """
        grad_batch = [p.grad_batch for p in params]

        if aggregate:
            grad_batch = torch.cat([g.flatten(start_dim=1) for g in grad_batch], dim=1)

        return grad_batch

    @staticmethod
    def _fetch_batch_l2_squared(params, aggregate=False):
        """Return individual gradient squared ℓ₂ norms, || ∇Lᵢ ||₂².

        Norms can be aggregated among parameters (summed). In this case, the return
        value has shape ``[N]`` for a mini-batch size of ``N``. Without aggregation,
        the parameter-wise individual squared gradient ℓ₂ norm is returned.

        Args:
            params ([torch.Tensor]): List of parameters whose individual gradient
                squared ℓ₂ norm will be fetched.
            aggregate (bool): Sum norms over parameter blocks.

        Returns:
            [torch.Tensor] if ``aggregate`` is False: List of same length as ``params``
                with tensors of shape ``[N]`` each.
            torch.Tensor if ``aggregate`` is True: A tensor of shape ``[N]``, containing
                the individual gradient squared norm.
        """
        batch_l2 = [p.batch_l2 for p in params]

        if aggregate:
            batch_l2 = sum(batch_l2)

        return batch_l2

    @staticmethod
    def _fetch_batch_dot(params, aggregate=False):
        """Return individual gradient pairwise dot products.

        Pairwise dot products can be aggregated (summed) among layers. In this case,
        the return value has shape ``[N, N]`` for a mini-batch size of ``N``. Without
        aggregation, the block-wise individual gradient pairwise dot products are
        returned.

        Args:
            params ([torch.Tensor]): List of parameters whose individual gradient
                pairwise dot products will be fetched.
            aggregate (bool): Sum dot products over parameter blocks.

        Returns:
            [torch.Tensor] if ``aggregate`` is False: List of same length as ``params``
                with tensors of shape ``[N, N]`` each.
            torch.Tensor if ``aggregate`` is True: A tensor of shape ``[N, N]``.
                containing pairwise dot products of aggregated individual gradients.
        """
        batch_dot_grad = [p.batch_dot for p in params]

        if aggregate:
            batch_dot_grad = sum(batch_dot_grad)

        return batch_dot_grad

    @staticmethod
    def _fetch_sum_grad_squared(params, aggregate=False):
        """Return sum of squared individual gradients.

        The sum of squared gradients can be aggregated (vectorized) among parameters.
        For a model with D parameters, the aggregated sum of squared gradients has
        shape ``[D]``. Without aggregation, the block-wise sum of squared individual
        gradients are returned.

        Args:
            params ([torch.Tensor]): List of parameters whose sum of squared individual
                gradients will be fetched.
            aggregate (bool): Concatenate (vectorize) results over parameter blocks.

        Returns:
            [torch.Tensor] if ``aggregate`` is False: List of same length as ``params``
                with items of same shape containing the gradient's second moment.
            torch.Tensor if ``aggregate`` is True: Gradient second moment of vectorized
                parameters.
        """
        sum_grad_squared = [p.sum_grad_squared for p in params]

        if aggregate:
            sum_grad_squared = torch.cat([sgs.flatten() for sgs in sum_grad_squared])

        return sum_grad_squared

    @staticmethod
    def _fetch_diag_curvature(params, savefield, aggregate=True):
        """Return diagonal curvature approximation.

        Diagonal curvature can be aggregated (vectorized) among parameters.
        For a model with D parameters, the aggregated diagonal curvature has
        shape ``[D]``. Without aggregation, the block-wise diagonal curvature
        is returned.

        Args:
            params ([torch.Tensor]): List of parameters whose diagonal curvature
                will be fetched.
            savefield (str): Field name where BackPACK stores the curvature
            aggregate (bool): Concatenate (vectorize) results over parameter blocks.

        Returns:
            [torch.Tensor] if ``aggregate`` is False: List of same length as ``params``
                with items of same shape containing the diagonal curvatures.
            torch.Tensor if ``aggregate`` is True: Diagonal curvature of vectorized
                parameters.
        """
        diag_curvature = [getattr(p, savefield) for p in params]

        if aggregate:
            diag_curvature = torch.cat([c.flatten() for c in diag_curvature])

        return diag_curvature

    def _fetch_batch_size_hotfix(self, batch_loss):
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
