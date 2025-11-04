class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], dropout=0.1):
        super(UNet, self).__init__()

        # ENCODER
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, dropout=dropout))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = feature

        # BOTTLENECK
        self.bottleneck = DoubleConv(features[-1], features[-1]*2, dropout=dropout)

        # DECODER
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.up_convs.append(DoubleConv(feature*2, feature, dropout=dropout))

        # OUTPUT
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # ENCODER PATH
        for i in range(len(self.downs)):
            x = self.downs[i](x)
            skip_connections.append(x)
            x = self.pools[i](x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # DECODER PATH
        for i in range(len(self.ups)):
            x = self.ups[i](x)
            skip_connection = skip_connections[i]

            # dopasuj rozmiar (na wypadek zaokrągleń)
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])

            x = torch.cat((skip_connection, x), dim=1)
            x = self.up_convs[i](x)

        return torch.sigmoid(self.final_conv(x))
