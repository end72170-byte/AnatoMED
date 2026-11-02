"""Causal role memory, candidate-dependent query, bounded residual and soft progression.

All inference inputs are frozen image embeddings/similarities, never annotations.
State is explicitly returned; callers reset it per clip and truncate gradients.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn

@dataclass
class State:
    memory: torch.Tensor
    available: torch.Tensor
    q: torch.Tensor
    sources: list
    confidences: list

    def detach(self):
        """Truncate recurrence without mutating prior support snapshots."""
        return State(self.memory.detach(),self.available.clone(),self.q.detach(),self.sources.copy(),self.confidences.copy())

class AnatomicalSearch(nn.Module):
    """Four-role query using five-class order eyes/upper/intact/lower/cleft."""
    def __init__(self, dim=128, gamma=.06, threshold=.20, gate=.15, beta=.5):
        super().__init__(); self.dim=dim; self.gamma=gamma;self.threshold=threshold;self.gate=gate;self.beta=beta
        self.memory_projection=nn.Linear(768,dim)
        self.role=nn.Embedding(4,dim);self.availability=nn.Embedding(2,dim)
        self.query=nn.Linear(768,dim);self.key=nn.Linear(dim,dim,bias=False);self.value=nn.Linear(dim,dim)
        self.residual=nn.Sequential(nn.Linear(dim*2+4,64),nn.GELU(),nn.Linear(64,1))
        self.transition_logits=nn.Parameter(torch.eye(4))

    def initial(self, device='cpu'):
        """Empty history and uniform acquisition state; new instance per clip."""
        return State(torch.zeros(4,768,device=device),torch.zeros(4,dtype=torch.bool,device=device),torch.full((4,),.25,device=device),[-1]*4,[-float('inf')]*4)

    def step(self,z,scores,state,frame_idx):
        """Read prior memory, score, snapshot, then write; returns differentiable score."""
        roles=torch.stack((scores[0],scores[1],scores[3],torch.maximum(scores[2],scores[4])))
        base=roles[3]; visual_gate=bool(base>=self.gate and base>=roles[:3].max())
        query=self.query(z)
        tokens=self.memory_projection(state.memory)+self.role.weight+self.availability(state.available.long())
        if bool(state.available.any()):
            weights=(self.key(tokens)@query/math.sqrt(self.dim)).masked_fill(~state.available,-1e9).softmax(0)
            context=weights@self.value(tokens)
            residual=self.gamma*torch.tanh(self.residual(torch.cat((query,context,state.q))).squeeze())
        else:
            weights=torch.zeros(4,device=z.device);context=query*0;residual=query.sum()*0
        final=base+residual
        accepted=visual_gate and bool(final>=self.threshold)
        prior=state.q@self.transition_logits.softmax(-1)
        q=(roles/.05+self.beta*torch.log(prior+1e-8)).softmax(0)
        # Fixed confidence writes; learned progression affects the subsequent query score.
        role=int(roles.argmax());confidence=float(roles[role]);write=(accepted if role==3 else confidence>=self.threshold and float(roles.topk(2).values.diff().abs()[0])>=.02)
        memory=state.memory.clone();available=state.available.clone();sources=state.sources.copy();confidences=state.confidences.copy()
        if write:
            g=float(torch.sigmoid((roles[role]-.20)/.05)) if bool(available[role]) else 1.0
            memory[role]=(1-g)*memory[role]+g*z; available[role]=True
            if confidence>confidences[role]: sources[role]=frame_idx;confidences[role]=confidence
        record=dict(base=base,residual=residual,final=final,gate=visual_gate,accepted=accepted,context=context,attention=weights,state_before=state.q,state_after=q,available_before=state.available.clone(),sources_before=state.sources.copy(),memory_before=state.memory.detach().clone())
        return record,State(memory,available,q,sources,confidences)
