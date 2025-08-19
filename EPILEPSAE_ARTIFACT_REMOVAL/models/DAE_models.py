import torch
import torch.nn as nn
import torch.nn.functional as F
import models.cnn_models as cnn_models



class Conv1DAutoencoder(nn.Module):
    def __init__(self, input_channels=19):
        super(Conv1DAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv1d(16, 4, kernel_size=3, padding="same"),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv1d(64, input_channels, kernel_size=3, padding="same"),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.permute(0, 2, 1) 
        return x
    

    def __init__(self, input_size, N_o=25, N_i=10, N_LS=32, num_layers=2):
        """
        LSTM Autoencoder for EEG signals.
        
        Parameters:
        - input_size: Number of EEG input features (channels)
        - N_o: Hidden size of the first LSTM layer
        - N_i: Hidden size of the second LSTM layer
        - N_LS: Latent space size (user-defined)
        - num_layers: Number of LSTM layers
        """
        super( LSTMAutoencoder, self).__init__()
        
        # Encoder: Two LSTM layers with No = 50 and Ni = 25
        self.encoder_lstm1 = nn.LSTM(input_size, N_o, num_layers=1, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(N_o, N_i, num_layers=1, batch_first=True)
        
        # Fully connected layer to project to latent space
        self.encoder_fc = nn.Linear(N_i, N_LS)
        
        # Decoder: Fully connected layer to expand back to Ni
        self.decoder_fc = nn.Linear(N_LS, N_i)
        
        # Decoder LSTM layers (reverse of encoder)
        self.decoder_lstm1 = nn.LSTM(N_i, N_o, num_layers=1, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(N_o, input_size, num_layers=1, batch_first=True)

    def forward(self, x):
        # Encoder
        window_size = x.shape[1]
        x, _ = self.encoder_lstm1(x)  # LSTM Layer 1 (No = 50)
        x, (hidden, _) = self.encoder_lstm2(x)  # LSTM Layer 2 (Ni = 25)
        
        # Flatten last hidden state and project to latent space
        latent = self.encoder_fc(hidden[-1])  # [batch_size, N_LS]
        
        # Decoder
        hidden_dec = self.decoder_fc(latent).unsqueeze(1)  # Expand to match LSTM input
        x, _ = self.decoder_lstm1(hidden_dec.expand(-1, window_size, -1)) 
        x, _ = self.decoder_lstm2(x) 
        
        return x


    def __init__(self, input_channels=19, seq_length=7680, N_o=50, N_i=25, N_LS=32):
        super(CNN_LSTM_Autoencoder, self).__init__()

        # 1️⃣ CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  

            nn.Conv1d(32, 64, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  
        )
        
        # Compute CNN output shape
        cnn_output_size = seq_length // 4  # Because of 2 MaxPools (each halves the size)

        # 2️⃣ LSTM Encoder
        self.encoder_lstm1 = nn.LSTM(64, N_o, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(N_o, N_i, batch_first=True)

        # 3️⃣ Fully Connected Latent Space
        self.encoder_fc = nn.Linear(N_i, N_LS)
        self.decoder_fc = nn.Linear(N_LS, N_i)

        # 4️⃣ LSTM Decoder (Reverse of Encoder)
        self.decoder_lstm1 = nn.LSTM(N_i, N_o, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(N_o, 64, batch_first=True)

        # 5️⃣ CNN Decoder
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            #nn.Upsample(scale_factor=2),
            
            nn.ConvTranspose1d(32, input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Normalize output between 0-1
        )

    def forward(self, x):
        window_size = x.shape[1]
        # CNN Feature Extraction (permute to match Conv1d: batch, channels, time)
        x = x.permute(0, 2, 1) 
        
        x = self.cnn(x)

        # LSTM expects (batch, seq_len, features) → Permute back
        x = x.permute(0, 2, 1)  

        # LSTM Encoder
       
        x, _ = self.encoder_lstm1(x)
        x, (hidden, _) = self.encoder_lstm2(x)

        # Latent space projection
        latent = self.encoder_fc(hidden[-1])

        # Decoder: Expand latent space and LSTM decode
        x = self.decoder_fc(latent).unsqueeze(1).expand(-1, window_size, -1)
        x, _ = self.decoder_lstm1(x)
        x, _ = self.decoder_lstm2(x)

        # CNN Decoder
        x = x.permute(0, 2, 1)  # Back to (batch, channels, time)
        
        x = self.deconv(x)
        x = x.permute(0, 2, 1) 
    
        return x


# VGG Block for UNet
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, ks,dropout=0.0):
        super().__init__()
        padding = int((ks - 1) / 2)

        self.relu = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, middle_channels, kernel_size=ks, padding=padding)
        self.bn1 = nn.BatchNorm1d(middle_channels)
        self.conv2 = nn.Conv1d(middle_channels, out_channels, kernel_size=ks, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        return out
  
# UNet model with VGG blocks with different depths
class UNet_3(nn.Module):
    def __init__(self, num_classes=2, input_channels=2,dropout=0, **kwargs):
        super().__init__()

        nb_filter = [16, 32, 64]

        self.pool = nn.MaxPool1d(2)
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)

      
        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0], ks=7,dropout=dropout)
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1], ks=5,dropout=dropout)
        
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2], ks=3,dropout=dropout)
        
        self.conv3_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1], ks=5,dropout=dropout)
        self.conv2_2 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0], ks=7,dropout=dropout)
 
        self.final = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1, padding=0)


    def forward(self, input):
        input = input.permute(0, 2, 1) 
        
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        
        x2_0 = self.conv2_0(self.pool(x1_0))
        
        x1_1 = self.conv3_1(torch.cat([x1_0, self.up(x2_0)], dim=1)) 
        x0_2 = self.conv2_2(torch.cat([x0_0, self.up(x1_1)], dim=1))

        output = self.final(x0_2)
        output = output.permute(0, 2, 1)
        return output
    
class UNet_4(nn.Module):
        def __init__(self, num_classes=2, input_channels=2,dropout=0, **kwargs):
            super().__init__()

            nb_filter = [16, 32, 64, 128]

            self.pool = nn.MaxPool1d(2)
          
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)

            
            self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0], ks=7,dropout=dropout)
            self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1], ks=5,dropout=dropout)
            self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2], ks=3,dropout=dropout)
            
            self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3], ks=3,dropout=dropout)
        
            self.conv3_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2], ks=3,dropout=dropout)
            self.conv2_2 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1], ks=5,dropout=dropout)
            self.conv1_3 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0], ks=7,dropout=dropout)
        

            self.final = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1, padding=0)


        def forward(self, input):
            input = input.permute(0, 2, 1) 
            x0_0 = self.conv0_0(input)
            x1_0 = self.conv1_0(self.pool(x0_0))
            x2_0 = self.conv2_0(self.pool(x1_0))
           
            x3_0 = self.conv3_0(self.pool(x2_0))
    

            x2_1 = self.conv3_1(torch.cat([x2_0, self.up(x3_0)], dim=1))  
            x1_2 = self.conv2_2(torch.cat([x1_0, self.up(x2_1)], dim=1))
            x0_3 = self.conv1_3(torch.cat([x0_0, self.up(x1_2)], dim=1))
       

            output = self.final(x0_3)
            output = output.permute(0, 2, 1)
            return output    

class UNet_5(nn.Module):
    def __init__(self, num_classes=2, input_channels=2,dropout=0, **kwargs):
        super().__init__()

        nb_filter = [16, 32, 64, 128, 256 ]

        self.pool = nn.MaxPool1d(2)
   
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0], ks=7,dropout=dropout)
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1], ks=5,dropout=dropout)
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2], ks=5,dropout=dropout)
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3], ks=3,dropout=dropout)
       
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4], ks=3,dropout= dropout)

        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3], ks=3,dropout=dropout)
        self.conv2_2 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2], ks=5,dropout=dropout)
        self.conv1_3 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1], ks=5,dropout=dropout)
        self.conv0_4 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0], ks=7,dropout=dropout)

        self.final = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1, padding=0)


    def forward(self, input):
        input = input.permute(0, 2, 1) 
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
       
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))  
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], dim=1))

        output = self.final(x0_4)
        output = output.permute(0, 2, 1)
        return output
    

# UNet models with Squeeze-and-Excitation ResBlocks
class SE_UNet_3(nn.Module):
    def __init__(self, num_classes=2, input_channels=2, **kwargs):
        super().__init__()

        nb_filter = [16, 32, 64]

        self.pool = nn.MaxPool1d(2)
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)

      
        self.conv0_0 = cnn_models.SE_ResBlock(input_channels, nb_filter[0],7)
        self.conv1_0 = cnn_models.SE_ResBlock(nb_filter[0], nb_filter[1],5)
        
        
        self.conv2_0 = cnn_models.SE_ResBlock(nb_filter[1], nb_filter[2], 3)
        
        self.conv3_1 = cnn_models.SE_ResBlock(nb_filter[1] + nb_filter[2], nb_filter[1],5)
        self.conv2_2 = cnn_models.SE_ResBlock(nb_filter[0] + nb_filter[1], nb_filter[0],7)
 
        self.final = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1, padding=0)


    def forward(self, input):
        input = input.permute(0, 2, 1) 
        
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        
        x2_0 = self.conv2_0(self.pool(x1_0))
        
        x1_1 = self.conv3_1(torch.cat([x1_0, self.up(x2_0)], dim=1)) 
        x0_2 = self.conv2_2(torch.cat([x0_0, self.up(x1_1)], dim=1))

        output = self.final(x0_2)
        output = output.permute(0, 2, 1)
        return output
    
class SE_UNet_4(nn.Module):
    def __init__(self, num_classes=2, input_channels=2,dropout=0, **kwargs):
        super().__init__()

        nb_filter = [16, 32, 64, 128]

        self.pool = nn.MaxPool1d(2)
        
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)

        
        self.conv0_0 = cnn_models.SE_ResBlock(input_channels, nb_filter[0],7)
        self.conv1_0 = cnn_models.SE_ResBlock(nb_filter[0], nb_filter[1], 5)
        self.conv2_0 = cnn_models.SE_ResBlock(nb_filter[1], nb_filter[2], 3)
        
        self.conv3_0 = cnn_models.SE_ResBlock(nb_filter[2], nb_filter[3], 3)
    
        self.conv3_1 = cnn_models.SE_ResBlock(nb_filter[2] + nb_filter[3], nb_filter[2], 3)
        self.conv2_2 = cnn_models.SE_ResBlock(nb_filter[1] + nb_filter[2], nb_filter[1], 5)
        self.conv1_3 = cnn_models.SE_ResBlock(nb_filter[0] + nb_filter[1], nb_filter[0], 7)
    

        self.final = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1, padding=0)


    def forward(self, input):
        input = input.permute(0, 2, 1) 
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        
        x3_0 = self.conv3_0(self.pool(x2_0))


        x2_1 = self.conv3_1(torch.cat([x2_0, self.up(x3_0)], dim=1))  
        x1_2 = self.conv2_2(torch.cat([x1_0, self.up(x2_1)], dim=1))
        x0_3 = self.conv1_3(torch.cat([x0_0, self.up(x1_2)], dim=1))
    

        output = self.final(x0_3)
        output = output.permute(0, 2, 1)
        return output   

class SE_UNet_5(nn.Module):
    def __init__(self, num_classes=2, input_channels=2,reduction=16,dropout=0.1, **kwargs):
        super().__init__()

        nb_filter = [16, 32, 64, 128, 256 ]

        self.pool = nn.MaxPool1d(2)
   
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)

        self.conv0_0 = cnn_models.SE_ResBlock(input_channels, nb_filter[0], 7,reduction=reduction)
        self.conv1_0 = cnn_models.SE_ResBlock(nb_filter[0], nb_filter[1], 5,reduction=reduction)
        self.conv2_0 = cnn_models.SE_ResBlock(nb_filter[1], nb_filter[2], 5,reduction=reduction)
        self.conv3_0 = cnn_models.SE_ResBlock(nb_filter[2], nb_filter[3], 3,reduction=reduction)
       
        self.conv4_0 = cnn_models.SE_ResBlock(nb_filter[3], nb_filter[4], 3,reduction=reduction)

        self.conv3_1 = cnn_models.SE_ResBlock(nb_filter[3] + nb_filter[4], nb_filter[3], 3,reduction=reduction)
        self.conv2_2 = cnn_models.SE_ResBlock(nb_filter[2] + nb_filter[3], nb_filter[2], 5,reduction=reduction)
        self.conv1_3 = cnn_models.SE_ResBlock(nb_filter[1] + nb_filter[2], nb_filter[1], 5,reduction=reduction)
        self.conv0_4 = cnn_models.SE_ResBlock(nb_filter[0] + nb_filter[1], nb_filter[0], 7,reduction=reduction)

        self.final = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1, padding=0)


    def forward(self, input):
        input = input.permute(0, 2, 1) 
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
       
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))  
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], dim=1))

        output = self.final(x0_4)
        output = output.permute(0, 2, 1)
        return output
    
class SE_UNet_6(nn.Module):
    def __init__(self, num_classes=2, input_channels=2,reduction=16,dropout=0, **kwargs):
        super().__init__()

        nb_filter = [16, 32, 64, 128, 256, 512]

        self.pool = nn.MaxPool1d(2)
   
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)

        self.conv0_0 = cnn_models.SE_ResBlock(input_channels, nb_filter[0], 7,reduction=reduction)
        self.conv1_0 = cnn_models.SE_ResBlock(nb_filter[0], nb_filter[1], 7,reduction=reduction)
        self.conv2_0 = cnn_models.SE_ResBlock(nb_filter[1], nb_filter[2], 5,reduction=reduction)
        self.conv3_0 = cnn_models.SE_ResBlock(nb_filter[2], nb_filter[3], 3,reduction=reduction)
        self.conv4_0 = cnn_models.SE_ResBlock(nb_filter[3], nb_filter[4], 3,reduction=reduction)
        
        self.conv5_0 = cnn_models.SE_ResBlock(nb_filter[4], nb_filter[5], 3,reduction=reduction)
        
        self.conv4_1 = cnn_models.SE_ResBlock(nb_filter[4] + nb_filter[5], nb_filter[4], 3,reduction=reduction)
        self.conv3_2 = cnn_models.SE_ResBlock(nb_filter[3] + nb_filter[4], nb_filter[3], 3,reduction=reduction)
        self.conv2_3 = cnn_models.SE_ResBlock(nb_filter[2] + nb_filter[3], nb_filter[2], 5,reduction=reduction)
        self.conv1_4 = cnn_models.SE_ResBlock(nb_filter[1] + nb_filter[2], nb_filter[1], 7,reduction=reduction)
        self.conv0_5 = cnn_models.SE_ResBlock(nb_filter[0] + nb_filter[1], nb_filter[0], 7,reduction=reduction)

        self.final = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1, padding=0)


    def forward(self, input):
        input = input.permute(0, 2, 1) 
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        
        x5_0 = self.conv5_0(self.pool(x4_0))
        
        x4_1 = self.conv4_1(torch.cat([x4_0, self.up(x5_0)], dim=1))
        x3_2 = self.conv3_2(torch.cat([x3_0, self.up(x4_1)], dim=1))  
        x2_3 = self.conv2_3(torch.cat([x2_0, self.up(x3_2)], dim=1))
        x1_4 = self.conv1_4(torch.cat([x1_0, self.up(x2_3)], dim=1))
        x0_5 = self.conv0_5(torch.cat([x0_0, self.up(x1_4)], dim=1))

        output = self.final(x0_5)
        output = output.permute(0, 2, 1)
        return output
    
class SE_UNet_7(nn.Module):
    def __init__(self, num_classes=2, input_channels=2,reduction=16,dropout=0, **kwargs):
        super().__init__()

        nb_filter = [16, 32, 64, 128, 256, 512,1024]

        self.pool = nn.MaxPool1d(2)
   
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)

        self.conv0_0 = cnn_models.SE_ResBlock(input_channels, nb_filter[0], 7,reduction=reduction)
        self.conv1_0 = cnn_models.SE_ResBlock(nb_filter[0], nb_filter[1], 7,reduction=reduction)
        self.conv2_0 = cnn_models.SE_ResBlock(nb_filter[1], nb_filter[2], 5,reduction=reduction)
        self.conv3_0 = cnn_models.SE_ResBlock(nb_filter[2], nb_filter[3], 5,reduction=reduction)
        self.conv4_0 = cnn_models.SE_ResBlock(nb_filter[3], nb_filter[4], 3,reduction=reduction)
        self.conv5_0 = cnn_models.SE_ResBlock(nb_filter[4], nb_filter[5], 3,reduction=reduction)
        
        self.conv6_0 = cnn_models.SE_ResBlock(nb_filter[5], nb_filter[6], 3,reduction=reduction)
        
        self.conv5_1 = cnn_models.SE_ResBlock(nb_filter[5] + nb_filter[6], nb_filter[5], 3,reduction=reduction)
        self.conv4_2 = cnn_models.SE_ResBlock(nb_filter[4] + nb_filter[5], nb_filter[4], 3,reduction=reduction)
        self.conv3_3 = cnn_models.SE_ResBlock(nb_filter[3] + nb_filter[4], nb_filter[3], 3,reduction=reduction)
        self.conv2_4 = cnn_models.SE_ResBlock(nb_filter[2] + nb_filter[3], nb_filter[2], 5,reduction=reduction)
        self.conv1_5 = cnn_models.SE_ResBlock(nb_filter[1] + nb_filter[2], nb_filter[1], 7,reduction=reduction)
        self.conv0_6 = cnn_models.SE_ResBlock(nb_filter[0] + nb_filter[1], nb_filter[0], 7,reduction=reduction)

        self.final = nn.Conv1d(nb_filter[0], num_classes, kernel_size=1, padding=0)


    def forward(self, input):
        input = input.permute(0, 2, 1) 
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x5_0 = self.conv5_0(self.pool(x4_0))
        
        x6_0 = self.conv6_0(self.pool(x5_0))
        
        x5_1= self.conv5_1(torch.cat([x5_0, self.up(x6_0)], dim=1))
        x4_2 = self.conv4_2(torch.cat([x4_0, self.up(x5_1)], dim=1))
        x3_3 = self.conv3_3(torch.cat([x3_0, self.up(x4_2)], dim=1))  
        x2_4 = self.conv2_4(torch.cat([x2_0, self.up(x3_3)], dim=1))
        x1_5 = self.conv1_5(torch.cat([x1_0, self.up(x2_4)], dim=1))
        x0_6 = self.conv0_6(torch.cat([x0_0, self.up(x1_5)], dim=1))

        output = self.final(x0_6)
        output = output.permute(0, 2, 1)
        return output
    
