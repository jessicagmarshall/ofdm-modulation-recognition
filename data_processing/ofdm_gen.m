function [  ] = ofdm_gen( k, nExamples )
%generates noiseless OFDM signal with an ifft of length numChannels
numChannels = 128;                             %or 64 pt IFFT
cycpre = 16;                                    %cyclic prefix
syms = randi([0,k-1], numChannels, nExamples);
msg = qammod(syms,k);
msgOFDM = ifft(msg, numChannels);
msgOFDM = [msgOFDM(end - cycpre:end, :) ; msgOFDM];
% msgOFDM is nChannels + cycpr by nExamples

msgR = real(msgOFDM);
msgI = imag(msgOFDM);

modStr = sprintf('ofdm%dqam', k);
for i = 1:nExamples
%       mod = randn(2,128,'single');
        mod = single([msgR(:,i); msgI(:,i)]);
        fName = sprintf('%difft%d_%d.bin', numChannels, k, i)                %change this to example for training data
        fid = fopen(fullfile('data_radio', modStr, fName), 'w');  %change this to data for training data
        fwrite(fid, mod, 'float32', 'ieee-le');
        fclose(fid);
end
end
