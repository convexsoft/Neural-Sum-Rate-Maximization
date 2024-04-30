clc;
clear;
w = [1;1;1];
% G = [1.5393    0.1952    0.0166;
%     0.0118    1.5271    0.0092;
%     0.1672    0.5377    1.7893];
G = [3.1929    0.1360    0.2379;
    0.0702    2.8835    0.2436;
    0.0693    0.0924    2.3060];
F = diag(diag(1./G))*(G-diag(diag(G)));
v = [1;1;1];
G_ndiag = G-diag(diag(G));
p_sum_max = 2;


% [p_bt,obj_bt] = B_tilde_alg(G, w, v, p_sum_max)
% 2.8266

% [p_mul,obj_mul,~] = multiplier_alg(G, w, v, p_sum_max)
% sum(p_mul)
% 2.8294

% [p_admm,obj_admm,~] = admm_alg(G, w, v, p_sum_max)


% [p,obj] = brutal_force(G, w, v, p_sum_max)
% sum(p)
% obj = 2.8390

num = 10000;
count = 0;
average = 0;
for i = 1:num
    p1 = rand(1);
    p2 = rand(1);
    p = [p1;p2;p_sum_max-p1-p2];
    sinr = diag(G).*p./(G_ndiag*p+1);
    obj = sum(log(1+sinr));
    if obj>=2.8350
        count=count+1;
    end

    average = (average*(i-1)+obj)/i;

end
average
count
count/num

function [p,obj,y] = multiplier_alg(G, w, v, p_sum_max)
    G_ndiag = G-diag(diag(G));
    y = rand(1);
    rho = 10e-3;
    p = rand(3,1);
    tol = 10e-8;
    err_y = 1;
    
    while err_y > tol
            err = 1;
            while err > tol
                p_temp = p;
                p = w./(y + rho*(sum(p)-p_sum_max) + G_ndiag'*(w./(G_ndiag*p+v)));
                err = norm(p-p_temp,2);
            end
            
            y_temp = y;
            y = y + rho*(sum(p)-p_sum_max);
            err_y = norm(y-y_temp,2);
        
    end
        sinr = diag(G).*p./(G_ndiag*p+1);
        obj = sum(log(1+sinr));
end

function [p_bt,obj_bt] = B_tilde_alg(G, w, v, p_sum_max)
    G_ndiag = G-diag(diag(G));

    F = diag(diag(1./G))*(G-diag(diag(G)));
    B = F + 1/p_sum_max*(ones(3,1)*v');
    B_tilte = inv(eye(3)+B)*B;
    
    cvx_begin gp
    variables z(3)
    minimize( prod((B_tilte*z).*inv_pos(z)) );
    subject to
        B_tilte*z<=z;
    cvx_end
    
    z = z/sum(z);
    p_bt = (eye(3)-B_tilte)*z;
    p_bt = p_bt/sum(p_bt)*2;

    sinr = diag(G).*p_bt./(G_ndiag*p_bt+1);
    obj_bt = sum(log(1+sinr));
end


function []=iter_z(G, w, v, p_sum_max)
    F = diag(diag(1./G))*(G-diag(diag(G)));
    B = F + 1/p_sum_max*(ones(3,1)*v');

    B_tilte = inv(eye(3)+B)*B
    tol = 10e-7;
    err = 1;
    z = rand(3,1);
    
    while err > tol
        z_temp = z;
        z = w./( B_tilte'*(w./(B_tilte*z)) );
        err = norm(z-z_temp,2);
    end

    z2 = z/sum(z)
    p = (eye(3)-B_tilte)*z
    p = inv(eye(3)+diag(v)*B)*z
    p = p/sum(p)*2

%     G_ndiag = G-diag(diag(G));
%     err2 = 1;
%     while err2 > tol
%         p_temp = p
%         sinr = diag(G).*p./(G_ndiag*p+1)
%         p = (sinr.*z)./(1+sinr)*p_sum_max/sum((eye(3)-B_tilte)*z);
%         err2 = norm(p_temp-p,2);
% 
% 
%     end
%     p = p/sum(p)*2
    
end



function [p,obj,y] = admm_alg(G, w, v, p_sum_max)
    G_ndiag = G-diag(diag(G));
    y = rand(3,1);
    mu = rand(3,1);
    rho = 10e-3;
    p = rand(3,1);
    tol = 10e-7;
    err_y = 1;
    p_len = length(p);
    
    while err_y > tol
            err = 1;
            while err > tol
                p_temp = p;
                p = w./(rho*(p-y+mu) + G_ndiag'*(w./(G_ndiag*p+v)));
                err = norm(p-p_temp,2);
            end
            y_temp = y;
            mu_temp = mu;
            y = (p+mu)-(sum(p+mu)-p_sum_max)/p_len;
            mu = mu+p-y;
            err_y = norm(y-y_temp,2)+norm(mu-mu_temp,2);
        
    end
        sinr = diag(G).*p./(G_ndiag*p+1);
        obj = sum(log(1+sinr));
end




function [p_opt,obj] = brutal_force(G, w, v, p_sum_max)
    G_ndiag = G-diag(diag(G));
    obj = -1;
    for p1 = 10e-3:10e-3:p_sum_max
        for p2 = 10e-3:10e-3:p_sum_max-p1
            for p3 = 10e-3:10e-3:p_sum_max-p1-p2
                p = [p1;p2;p3];
                 sinr = diag(G).*p./(G_ndiag*p+1);
                 if sum(log(1+sinr))>=obj
                    p_opt = p;
                    obj = sum(log(1+sinr));  
                 end

            end

        end

    end

end






