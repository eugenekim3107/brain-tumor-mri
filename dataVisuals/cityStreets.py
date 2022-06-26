# Function to display the cumulative streets of a specified city

def city_map(city, cell_width, vmax, cmap):
    dataset = ArgoverseDataset(city=city, split="train")
    data_loader = DataLoader(dataset, batch_size=len(dataset))
    for sample in data_loader:
        inp, out = sample
    inp = inp.reshape(-1,2)
    out = out.reshape(-1,2)
    complete = torch.concat((inp,out), axis=0)
    X = complete[:,0] - complete[:,0].min()
    Y = complete[:,1] - complete[:,1].min()
    final = np.empty((int(X.max()/cell_width) + 1, int(Y.max()/cell_width) + 1))
    for x, y in zip(X,Y):
        temp_x = int(torch.div(x, cell_width))
        temp_y = int(torch.div(y, cell_width))
        final[temp_x, temp_y] += 1
    return sns.heatmap(final, vmin=0, vmax=vmax, cmap=cmap)

# Example showing the streets of Austin
city_map("austin", 50, 200, "YlOrBr")
