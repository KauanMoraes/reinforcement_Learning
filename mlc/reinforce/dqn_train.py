import argparse
import random
import re
from collections import deque

import ale_py
import gymnasium as gym
import numpy as np
import torch
import random
import os
from torch import nn 
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from mlc.command.base import Base
from mlc.reinforce.car_nets import ModeloDQN as Modelo
from mlc.reinforce.memory import MultistepReplayBuffer, PriorityQueue
from mlc.util.resources import get_time_as_str


class TrainDQN(Base):

    def __init__(self, hparams):
        super().__init__(hparams)

        # try to use the device specified in the arguments
        self.device = "cpu"
        if hparams["device"].startswith("cuda"):
            if torch.cuda.is_available():
                self.device = torch.device(hparams["device"])
                print(f"Using device: {self.device}")
                print(f"CUDA Version: {torch.version.cuda}")
                print(f"Device Name: {torch.cuda.get_device_name(0)}")
            else:
                raise RuntimeError("CUDA is not available")
        self.hparams = hparams
        self.gamma = hparams["gamma"]
        # MODIFICADO: DQN funciona com ações discretas
        self.mode = "discrete" 
        if hparams["mode"] == "continuous":
            print("AVISO: DQN clássico não suporta ações contínuas. Usando modo 'discrete'.")
        
        # latest_link_path = f"{agent_folder}/latest"
        # # Remove o link antigo, se existir (lexists é seguro para links quebrados)
        # if os.path.lexists(latest_link_path):
        #     os.remove(latest_link_path)
        # # Cria um novo link simbólico apontando para a pasta da execução atual
        # os.symlink(run_name, latest_link_path)
        # print(f"Link 'latest' criado, apontando para: {self.output_folder}")
            
        self.num_stack = hparams["num_stack"]
        self.frame_skip = hparams["frame_skip"]
        self.lr_decay = hparams["lr_decay"]
        self.learning_rate = hparams["learning_rate"]
        if hparams["name"]:
            self.output_folder = f"agents/{hparams['game'].replace('/', '_')}/multi/{hparams['name']}"
        else:
            self.output_folder = f"agents/{hparams['game'].replace('/', '_')}/multi/{get_time_as_str()}"
        os.makedirs(f"{self.output_folder}/checkpoints", exist_ok=True)
        self.writer = SummaryWriter(self.output_folder + "/tensorboard")
        #gym.register(ale_py)
        
        # ADICIONADO: Hiperparâmetros específicos do DQN
        self.batch_size = hparams["batch_size"]
        self.epsilon_start = hparams["epsilon_start"]
        self.max_episodes = hparams["max_episodes"]
        self.epsilon = self.epsilon_start
        self.kf=0
        self.grtst_reward = (None,-np.inf)
        self.epsilon_end = hparams["epsilon_end"]
        self.epsilon_decay = hparams["epsilon_decay"]
        self.target_update_freq = hparams["target_update"]
        self.max_grad_norm = hparams["max_grad_norm"]
        self.capacity = self.hparams["buffer_size"]
        self.n_step = 5
        self.gamma = self.hparams["gamma"]
        self.memory = MultistepReplayBuffer(
                            capacity=self.capacity,
                            n_step=self.n_step,  
                            gamma=self.gamma
        )
        self.memory_switched = False

    @classmethod
    def name(cls):
        return "traindqn_car"

    @staticmethod
    def add_arguments(parser):
        def _parse_device_arg(arg_value):
            pattern = re.compile(r"(cpu|cuda|cuda:\d+)")
            if not pattern.match(arg_value):
                raise argparse.ArgumentTypeError("invalid value")
            return arg_value

        parser.add_argument("-s", "--seed", type=int, default=42)
        parser.add_argument("-e", "--max_episodes", type=int, default=2000)

        parser.add_argument("-g", "--game", default="CarRacing-v3")
        parser.add_argument("--num_env", default=1, type=int)
        parser.add_argument("-d", "--device", type=_parse_device_arg, default="cuda", help="device to use for training")
        parser.add_argument("-l", "--learning-rate", type=float, default=1e-4, help="learning rate for the optimizer")
        parser.add_argument("-c", "--check-point", type=int, default=20, help="check point every n episodes")
        parser.add_argument("--resume-from", type=str, default=None, help="path to checkpoint to resume training from")
        parser.add_argument("-v", "--video", type=int, default=20, help="create a video every n episodes") 
        parser.add_argument("-n", "--name", type=str, default=None, help="name this run")
        parser.add_argument("--gamma", type=float, default=0.99, help="discount factor for rewards")
        # O modo é fixado para discreto, mas o argumento é mantido para compatibilidade
        parser.add_argument("--mode", type=str, default="discrete", choices=["discrete", "continuous"], help="mode of the agent")
        parser.add_argument("--lr_decay", default=False, action="store_true", help="enable learning rate decay")
        parser.add_argument("--frame_skip", type=int, default=4, help="number of frames to skip for the agent input")
        parser.add_argument("--num_stack", type=int, default=4, help="number of frames to stack for the agent input")
        parser.add_argument("--validation", type=str, default=None, help="path to validation script (not used in training)")
        
        # Argumentos para DQN
        parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size for training")
        parser.add_argument("--buffer-size", type=int, default=15000, help="size of the replay buffer")
        parser.add_argument("--epsilon-start", type=float, default=1, help="starting value of epsilon")
        parser.add_argument("--epsilon-end", type=float, default=0.05, help="final value of epsilon")
        parser.add_argument("--epsilon-decay", type=float, default=50000, help="epsilon decay rate") # quanto menor, maior a velocidade de decaimento
        parser.add_argument("--target-update", type=int, default=20, help="frequency of target network updates")#### mudar p steps? rede aprendendo a morrer rápido p diminuir a dif ?
        parser.add_argument("--learning-starts", type=int, default=5000, help="number of steps before starting training")
        parser.add_argument("--max-steps", type=int, default=1000, help="maximum number of steps per episode")
        parser.add_argument("--max-grad-norm", type =float,default = 2.0, help = "max norm of the gradient")

    # Função para selecionar ação com epsilon-greedy
    def decay_epsilon(self, steps_done):
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * steps_done / self.epsilon_decay)   
    
    def select_action(self, state: torch.Tensor, policy_net, n_action, steps_done):
        action = []
        # Para cada ambiente no vetor
        if random.random() > self.epsilon :
            with torch.no_grad():
                # Pega o valor Q para o estado do ambiente i
                policy_net.eval() # Modo de avaliação
                # Late casting: Move para GPU e converte para float32 escalado
                state = state.to(self.device, dtype=torch.float32) / 255.0
                q_values = policy_net(state)
                # Escolhe a ação com maior valor Q
                action = q_values.max(1)[1] # Índices das ações com maior Q-value 
                action = action.item()
                policy_net.train() # Volta para o modo de treinamento
        else:
            action = random.randrange(n_action)
        return action
    
    # Função para otimizar o modelo (fazer o update do DQN)
    def optimize_model(self, policy_net: Modelo, target_net: Modelo, optimizer):
        if len(self.memory) < self.batch_size:
            return None # Não treina se o buffer não tiver amostras suficientes

        batch = self.memory.sample(self.batch_size)

        state_batch = torch.stack(batch[0]).to(self.device, dtype=torch.float32) / 255.0

        action_batch = torch.tensor(batch[1], dtype=torch.int64, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32, device=self.device) # soma dos rewards após ns passos

        next_state_batch = torch.stack(batch[3]).to(self.device, dtype=torch.float32) / 255.0  # estado após ns passos
        termination_batch = torch.tensor(batch[4], dtype=torch.float32, device=self.device)

        ns_batch = torch.tensor(batch[5], dtype=torch.int64).to(self.device) # how many steps to look ahead

        if isinstance(self.memory, PriorityQueue):
            indices= torch.tensor(batch[6], dtype=torch.int64, device=self.device)
            weights = torch.tensor(batch[7], dtype=torch.float32, device=self.device)
        else:
            indices = None
            weights = np.ones_like(reward_batch)
        
        #with torch.autocast(device_type=self.device):
        # Calcula Q(s_t, a), e então selecionamos as colunas das ações tomadas
        # O valor estimado pela policy_net no estado atual
        q_values = policy_net(state_batch).gather(1, action_batch) # dim = n_batch x 1

        with torch.no_grad():
            # A policy_net escolhe a melhor ação para o próximo estado ns passos a frente
            best_next_actions = policy_net(next_state_batch).max(1)[1].unsqueeze(1) 
            # [(max_value,indices)] of dim = n_batch, que transforma em ([best_next_actions]) de dim = n_batch x 1
            
            # A target_net avalia o valor dessa ação escolhida
            next_q_values = target_net(next_state_batch).gather(1, best_next_actions).squeeze(1)
            # O valor do próximo estado é 0 se o episódio terminou.
            next_q_values[termination_batch.bool()] = 0.0
        
        # Calcula a estimativa target olhando ns_batch+1 passos a frente
        target_q_values = reward_batch + (self.gamma ** (ns_batch + 1) * next_q_values)

        # Calcula a Loss (MSE)
        if indices is not None:
            weights = weights.unsqueeze(1)

            criterion = nn.SmoothL1Loss(reduction = 'none')
            elementwise_loss = criterion(q_values, target_q_values.unsqueeze(1))
            loss = torch.mean(elementwise_loss*weights)
            td_errors= elementwise_loss.detach().cpu().squeeze().numpy()
            self.memory.update_priorities(indices, td_errors)
        else:
            criterion = nn.SmoothL1Loss()
            loss = criterion(q_values, target_q_values.unsqueeze(1))

        # Otimiza o modelo
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=self.max_grad_norm)
        optimizer.step()
        return loss.item()
    
    #@profile # teria q rodar: kernprof -l -v .venv/bin/mlc --config mlc/config/params.yaml traindqn_car
    def run(self):
        #num_env = self.hparams["num_env"] 
        device = self.device
        self.memory.clear() # Limpa o buffer de memória antes de começar
        env = gym.make("CarRacing-v3", lap_complete_percent=0.95, #render_mode="rgb_array",
                             domain_randomize=False, continuous=False)

        # n_action = env.single_action_space.n se fosse vetorizado
        n_action = env.action_space.n 

        # Criação das duas redes: policy e target
        # A classe Modelo deve retornar Q-values (sem softmax no final)
        policy_net = Modelo(dim_hidden=64, init_ch=3*self.num_stack, dim_out=n_action).to(device)
        target_net = Modelo(dim_hidden=64, init_ch=3*self.num_stack, dim_out=n_action).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        learning_rate = self.hparams["learning_rate"]
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
        
        # Inicializa o GradScaler para o AMP (se estiver rodando na GPU)
        # scaler = torch.cuda.amp.GradScaler() if self.device == "cuda" else None
        
        episode_start = 0
        step = 0

        if self.hparams["resume_from"] and os.path.exists(self.hparams["resume_from"]):
            print(f"Retomando treinamento do checkpoint: {self.hparams['resume_from']}")
            checkpoint = torch.load(self.hparams["resume_from"], map_location=device)

            policy_net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Sincroniza a target_net com a policy_net carregada
            target_net.load_state_dict(policy_net.state_dict())
            
            episode_start = checkpoint['episode']

            
            print(f"Checkpoint carregado. Começando do episódio {episode_start}.")
               
        def process_obs(obs): # Utilizada na simulação 
            # Mantém em uint8 na CPU para economizar RAM
            obs_tensor = torch.tensor(np.array(obs), dtype=torch.uint8)
            H, W, C = obs_tensor.shape
            return obs_tensor.permute(2, 0, 1) # 3, 96, 96

        # Loop de treinamento principal
        print("Iniciando o treinamento...")
        pbar = tqdm(total=self.max_episodes,initial=episode_start, desc="Episódios concluídos")
                
        # Loop baseado em passos (steps) 
        for episode in range(episode_start, self.max_episodes):
            self.decay_epsilon(step)
            
            state, _ = env.reset(options={"randomize": False})
            state = process_obs(state) # Permanece na CPU
            episode_rewards = 0.0 
            episode_frames = [] 
            # No-op
            for _ in range(50):
                state, _, terminated, truncated, _ = env.step(0)
                if terminated or truncated:
                    break
            state = process_obs(state) # Permanece na CPU

            frame_buffer = deque([state]*self.num_stack,maxlen=self.num_stack)
            stacked_state = torch.cat(list(frame_buffer), dim = 0) # Permanece na CPU
            for time in range(self.hparams["max_steps"]): # Loop dentro do episódio
                step += 1
                # Seleciona a ação usando epsilon-greedy
                if episode < 5 and time < 1000:
                    action = random.randrange(1,3+1) 
                else:
                    action = self.select_action(stacked_state.unsqueeze(0), policy_net, n_action, step)
                                
                # Executa a ação no ambiente
                total_shaped_reward = 0
                total_env_reward = 0
                # Frame Skipping: executa a ação várias vezes para acelerar o jogo
                for i in range(self.frame_skip):
                    next_obs, reward, terminations, truncations, _ = env.step(action)                
                    total_env_reward += reward
                    
                    # Ações discretas padrão: 0: nada, 1: esquerda, 2: direita, 3: acelerar, 4: frear
                    shaped_reward = reward
                    if action == 0:
                        shaped_reward -= 0.1   # Penalidade severa por não fazer nada
                    # elif action == 3:
                    #     shaped_reward += 0.05  # Incentivo leve para manter a aceleração
                    # elif action == 4:
                    #     shaped_reward -= 0.05  # Penalidade leve por frear desnecessariamente
                    
                    total_shaped_reward += shaped_reward

                    if terminations or truncations:
                        break    
                next_state = process_obs(next_obs) # Estado após o frame skipping
                frame_buffer.append(next_state) # Atualiza o buffer de frames
                stacked_next_state = torch.cat(list(frame_buffer), dim = 0) # (C*num_stack, H, W) na CPU

                episode_frames.append(next_obs)               
                dones = np.logical_or(terminations, truncations)
                
                # Armazena as transições no replay buffer em uint8
                self.memory.store(
                    stacked_state, 
                    action, 
                    total_shaped_reward/10, # Reward Scaling 
                    stacked_next_state,
                    dones
                )
                episode_rewards += total_env_reward
                stacked_state = stacked_next_state
                # Se um episódio terminou
                if dones:
                    pbar.update(1)
                    self.writer.add_scalar("reward", episode_rewards, episode)
                    if episode_rewards>= self.grtst_reward[1]: self.grtst_reward= (episode,episode_rewards) 
                    episode_rewards = 0.0 # Reseta a recompensa do episódio
                                
                    if episode>=self.max_episodes/3 and episode % self.hparams["video"] == 0:
                        # Converte a lista de frames (T, H, W, C) para um tensor (N, T, C, H, W)
                        video_array = np.array(episode_frames, dtype=np.uint8).transpose(0, 3, 1, 2)
                        vid_tensor = torch.from_numpy(video_array).unsqueeze(0)                       
                        self.writer.add_video("gameplay", vid_tensor, global_step=episode, fps=30)                                                   
                    break
                                                   
            if self.kf ==0 and step >= self.hparams["learning_starts"]:
                print(f"Passo {step}, Episódios concluídos: {episode}, Epsilon: {self.epsilon:.4f}")
                self.kf = 1
            if episode==7 and not self.memory_switched:
                new_memory = PriorityQueue(capacity = self.capacity,n_step=self.n_step,gamma=self.gamma)
                # Copy old data over (optional)
                for item in self.memory.buffer:
                    new_memory.tree.add(1.0, item)  # set default priority
                self.memory = new_memory
                self.memory_switched = True
                print("🧠 Switched to Prioritized Replay Buffer.")
            # Treina a rede
            if step > self.hparams["learning_starts"]:

                loss = self.optimize_model(policy_net, target_net, optimizer)#, scaler=scaler)               
                if loss is not None:
                    self.writer.add_scalar("loss", loss, episode)
            self.writer.add_scalar("hyperparameters/epsilon", self.epsilon, episode)
            self.writer.flush()
            # Atualiza a target network
            if episode % self.target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())
                
            # Checkpoint do modelo
            if episode>self.max_episodes/3 and episode % self.hparams["check_point"] == 0: 
                checkpoint_path = f'{self.output_folder}/checkpoints/{episode:06d}.pt'
                torch.save({
                    'episode': episode,
                    'step': step,
                    'model_state_dict': policy_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)

        pbar.close()
        print("Treinamento concluído.")
        torch.save(policy_net.state_dict(), f'{self.output_folder}/final_model.pt')
        self.writer.close()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    TrainDQN.add_arguments(parser)

    args = parser.parse_args()
    hparams = vars(args)
    t = TrainDQN(hparams) 
    t.run()
