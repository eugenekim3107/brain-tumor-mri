# Function to display input and output of car trajectory

def example_traj(num_samples):
    for sample in train_data:
        inp, out = sample
        print("Input Shape: " + str(inp[0].shape), "\nOutput Shape: " + str(out[0].shape))
        for i in range(num_samples):
            plt.scatter(x=inp[i,:,0], y=inp[i,:,1], c="black", label="Input", alpha=0.5)
            plt.scatter(x=out[i,:,0], y=out[i,:,1], c="red", label="Output", alpha=0.5)
            plt.legend()
            plt.title(f"Example {i + 1} of Vehicle Trajectory")
            plt.show()
        break

# Example showing 5 input and output points of car trajectory
example_traj(5)
