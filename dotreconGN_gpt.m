function [x, logpost] = dotreconGN_gpt(y, x0, geom, hreg, niter)

    hMesh = geom.hMesh;
    hBasis = geom.hBasis; % toastBasis object
    n = geom.n;
    nb = hBasis.Count();
    frq = geom.frq;
    c0 = geom.c0;

    qvec = geom.qvec;
    mvec = geom.mvec;

    Le = hreg.Le;
    H = hreg.H;
    xprior = hreg.x;

    x = x0;
    logpost = zeros(niter, 1);

    for it = 1:niter
        % --- Map from basis to full mesh ---
        mua = hBasis.Map(x(1:nb));
        mus = hBasis.Map(x(nb+1:end));

        % --- Forward model ---
        K = dotSysmat(hMesh, mua, mus, geom.ref, frq);
        phi = K \ qvec;
        gamma = mvec.' * phi;
        ymod = [real(log(gamma(:))); imag(log(gamma(:)))];

        % --- Residual ---
        r = y - ymod;

        % --- Jacobians ---
        dK_dmua = dotJacobianmua(hMesh, mua, mus, geom.ref, frq, phi);
        dK_dmus = dotJacobianmus(hMesh, mua, mus, geom.ref, frq, phi);

        % Derivatives of gamma w.r.t mua and mus
        dgamma_dmua = mvec.' * (K \ (-dK_dmua * (K \ qvec)));
        dgamma_dmus = mvec.' * (K \ (-dK_dmus * (K \ qvec)));

        % Log transform derivative
        D1 = diag(1 ./ gamma(:));
        dymod_dmua = real(D1 * dgamma_dmua);
        dymod_dmus = real(D1 * dgamma_dmus);
        dymod_dmua_im = imag(D1 * dgamma_dmua);
        dymod_dmus_im = imag(D1 * dgamma_dmus);

        % Full Jacobian (2m × 2n)
        J_full = [dymod_dmua; dymod_dmua_im, dymod_dmus; dymod_dmus_im];

        % --- Map Jacobians into basis space ---
        B = hBasis.Matrix();  % (n × nb) basis matrix
        J_mua_b = J_full(:, 1:n) * B;
        J_mus_b = J_full(:, n+1:end) * B;

        J = [J_mua_b, J_mus_b];  % (2m × 2nb)

        % --- Gauss-Newton step ---
        A = J.' * Le.' * Le * J + H;
        b = J.' * Le.' * Le * r + H * (xprior - x);

        dx = A \ b;
        x = x + dx;

        logpost(it) = -0.5 * (r' * Le' * Le * r + (x - xprior)' * H * (x - xprior));

        fprintf('Iteration %d: log posterior = %.4f\n', it, logpost(it));
    end
end
