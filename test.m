M1 = randn(2000*10,4096, 'single');
M2 = randn(2000*10,4096, 'single');

th = tic();
sim = M1 * M2';
fprintf('cpu took %f secs\n', toc(th));

th = tic();
gM1 = gpuArray(M1);
gM2 = gpuArray(M2');
fprintf('xfer took %f secs\n', toc(th));
sim1 = gM1 * gM2;
fprintf('after mult before gather %f secs\n', toc(th));
th1 = tic();
res = gather(sim1);
fprintf('gather took %f secs\n', toc(th1));
fprintf('gpu took %f secs\n', toc(th));