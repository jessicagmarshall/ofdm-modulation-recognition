function [  ] = ofdm_gen_noisy( k, nExamples, SNR)

%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
numChannels = 128;                              %or 64 pt IFFT
cycpre = 16;                                    %cyclic prefix
syms = randi([0,k-1], numChannels, nExamples);
msg = qammod(syms,k);
msgNoisy = awgn(msg, SNR, 'measured');
msgOFDM = ifft(msgNoisy, numChannels);
msgOFDM = [msgOFDM(end - cycpre:end, :) ; msgOFDM];
% msgOFDM is nChannels + cycpr by nExamples

msgR = real(msgOFDM);
msgI = imag(msgOFDM);

modStr = sprintf('ofdm%dqam', k);
for i = 1:nExamples
%       mod = randn(2,128,'single');
        mod = single([msgR(:,i); msgI(:,i)]);
        fName = sprintf('test_0dB_SNR_%d.bin', i)       %change 20 to SNR value, example to test
        fid = fopen(fullfile('test_0dB_SNR', modStr, fName), 'w'); %change 20 to SNR value, data to test
        fwrite(fid, mod, 'float32', 'ieee-le');
        fclose(fid);
end
end
