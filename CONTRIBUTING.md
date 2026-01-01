# Contributing to UHS ConvLSTM-UNet Surrogate

Thank you for your interest in contributing to this research project!

## How to Contribute

### Reporting Issues

If you find bugs, have questions about the methodology, or want to suggest improvements:

1. Check if the issue already exists in the [Issues](../../issues) section
2. If not, create a new issue with:
   - Clear description of the problem/question
   - Steps to reproduce (for bugs)
   - Your environment details (OS, Python version, PyTorch version)
   - Relevant code snippets or error messages

### Code Contributions

We welcome contributions in the following areas:

#### 1. Model Improvements
- Alternative architectures (e.g., Transformers, Graph Neural Networks)
- Enhanced physics-informed loss functions
- Better hysteresis modeling approaches

#### 2. Training Strategies
- Advanced scheduled sampling variants
- Curriculum learning approaches
- Multi-task learning formulations

#### 3. Evaluation Tools
- Additional metrics and visualizations
- Uncertainty quantification methods
- Physical consistency checks

#### 4. Documentation
- Tutorial notebooks
- Code documentation improvements
- Use case examples

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature-name`)
3. Make your changes following our coding standards (see below)
4. Add tests if applicable
5. Update documentation as needed
6. Commit with clear, descriptive messages
7. Push to your fork
8. Submit a pull request with:
   - Description of changes
   - Motivation and context
   - Test results showing your changes work

## Coding Standards

### Python Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use docstrings for all functions and classes

### Example:
```python
def predict_reservoir_state(
    model: ConvLSTMUNet,
    initial_state: torch.Tensor,
    num_steps: int = 33
) -> torch.Tensor:
    """
    Perform autoregressive rollout for reservoir prediction.

    Args:
        model: Trained ConvLSTM-UNet model
        initial_state: Initial reservoir state, shape (B, 5, Nx, Ny, Nz)
        num_steps: Number of timesteps to predict

    Returns:
        Predicted states, shape (B, num_steps, 3, Nx, Ny, Nz)
    """
    # Implementation here
    pass
```

### Code Organization
- Keep functions focused and modular
- Separate concerns (data loading, model definition, training, evaluation)
- Use meaningful variable names
- Add comments for complex logic

### Testing
- Add unit tests for new functionality
- Ensure existing tests pass
- Test on both medium and high-fidelity data

## Research Collaboration

If you're interested in extending this work for academic research:

1. **Using the model**: Feel free to use and adapt the code. Please cite the original paper.

2. **Collaborative projects**: Contact Dr. Narjisse Kabbaj (nkabbaj@effatuniversity.edu.sa) to discuss potential collaborations.

3. **Data sharing**: If you have relevant UHS experimental or simulation data, we'd love to hear about potential data-sharing opportunities.

## Questions?

For questions about:
- **Code/Implementation**: Open an issue on GitHub
- **Methodology**: Refer to the paper or contact the author
- **Collaboration**: Email nkabbaj@effatuniversity.edu.sa

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for helping improve this research!
