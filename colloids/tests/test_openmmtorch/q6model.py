import torch
from rsh import rsh_cart_6


def switch_gauss(r, r0: float, d0: float):
    x = torch.square(r-d0)/(2*r0**2)
    return torch.exp(-x)

def calc_stein_single(i: int, pos, num_nbs: int, order: int, r0: float, d0: float, device: torch.device):
    disps = pos - pos[i]
    dists = torch.square(disps)
    dists = torch.sum(dists, dim=(1))
    dists = torch.sqrt(dists)
    
    inds_nbs = torch.argsort(dists)[1:num_nbs+1]
    disps = pos[inds_nbs] - pos[i]
    dists = torch.sqrt(torch.sum(torch.square(disps), dim=(1)))
    disps_nbs = disps.T/dists
    disps_nbs = disps_nbs.T
    
    Y_nm_real = rsh_cart_6(disps_nbs, device)

    sigma = switch_gauss(dists, r0, d0)
    sigma_sum = torch.sum(sigma)
    q_nm_real = torch.sum(sigma[:,None]*Y_nm_real[:,order*(order+1)-order:order*(order+1)+order+1], dim=(0))/sigma_sum
    
    return q_nm_real

def calc_stein(pos, num_nbs: int, order: int, r0: float, d0: float, device: torch.device):
    natoms = len(pos)

    q_nm_i = calc_stein_single(0, pos, num_nbs, order, r0, d0, device)
    q_n = torch.sqrt(torch.sum(q_nm_i*q_nm_i))
    for i in range(1, natoms):
        q_nm_i = calc_stein_single(i, pos, num_nbs, order, r0, d0, device)
        q_n_i = torch.sqrt(torch.sum(q_nm_i*q_nm_i))
        q_n = q_n + q_n_i
    return q_n/natoms

class Q6Module(torch.nn.Module):
    def __init__(self, num_nbs, order, r0, d0):
        super().__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.num_nbs = num_nbs
        self.order = order
        self.r0 = r0
        self.d0 = d0

    def forward(self, positions):
        positions = positions.float()
        q_n = calc_stein(positions, self.num_nbs, self.order,
                         self.r0, self.d0, self.device)
        return q_n


def main():
    num_nbs = 64
    q_order = 6
    r0 = 50.0
    d0 = 200.0
    cvmodule = torch.jit.script(Q6Module(num_nbs, q_order, r0, d0))    
    cvmodule.save('model.pt')

if __name__ == '__main__':
    main()
