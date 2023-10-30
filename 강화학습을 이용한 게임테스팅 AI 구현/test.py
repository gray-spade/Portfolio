import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import pandas as pd
from absl import app
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions
from pysc2.lib import features


MAX_TARGET = 21
_NA =MAX_TARGET
_NS =_NA*3
MAPNAME = 'CollectMineralShards'
APM = 16
UNLIMIT = 0 #초당 스탭 제한
VISUALIZE = False #파이게임으로 시각화 할것인가  False시 작동속도 증가
REALTIME = False  #실시간 작동을 할 것인가 False 면 빠르게 작동
SCREEN_SIZE = 84 #화면 크기
MINIMAP_SIZE = 64 #미니맵 크기
MOVE_SCREEN = 331 #이동 명령
NOT_QUEUED = [0]
players = [sc2_env.Agent(sc2_env.Race.terran)]

interface = features.AgentInterfaceFormat( \
    feature_dimensions=features.Dimensions( \
        screen=SCREEN_SIZE, minimap=MINIMAP_SIZE), use_feature_units=True)
# 재현 버퍼 (Replay Buffer)
class ReplayBuffer():
    def __init__(self,
            max_size=10000, # 버퍼 최대 크기
            batch_size=64): # 배치 크기
        # 저장되는 경험 튜플 초기화
        self.ss_mem = np.empty(shape=(max_size), dtype=np.ndarray) # 상태
        self.as_mem = np.empty(shape=(max_size), dtype=np.ndarray) # 행동
        self.rs_mem = np.empty(shape=(max_size), dtype=np.ndarray) # 보상
        self.ps_mem = np.empty(shape=(max_size), dtype=np.ndarray) # 다음 상태
        self.ds_mem = np.empty(shape=(max_size), dtype=np.ndarray) # 종료 여부
        # 변수 초기화
        self.max_size = max_size # 최대 크기
        self.batch_size = batch_size # 배치 크기
        self._idx = 0 # 인덱스
        self.size = 0 # 버퍼 크기

    # 새로운 샘플 저장
    def store(self, sample):

        # 경험 튜풀 샘플 저장
        s, a, r, p, d = sample  # 경험 튜플
        self.ss_mem[self._idx] = s  # 상태
        self.as_mem[self._idx] = a  # 행동
        self.rs_mem[self._idx] = r  # 보상
        self.ps_mem[self._idx] = p  # 다음 상태
        self.ds_mem[self._idx] = d  # 종료 여부
        # 인덱스 증가
        self._idx += 1
        self._idx = self._idx % self.max_size
        # 버퍼 크기 증가
        self.size += 1
        self.size = min(self.size, self.max_size)

    # 배치 크기로 샘플링
    def sample(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        # 배치 크기 단위로 id 랜덤 선택
        idxs = np.random.choice(self.size, batch_size,replace=False)
        # 샘플링된 id로 재현 버퍼에서 경험 튜플 추출
        experiences = np.vstack(self.ss_mem[idxs]),\
                      np.vstack(self.as_mem[idxs]), \
                      np.vstack(self.rs_mem[idxs]), \
                      np.vstack(self.ps_mem[idxs]), \
                      np.vstack(self.ds_mem[idxs])
        return experiences
# 버퍼의 크기를 리턴, 매직 메서드
    def __len__(self):
        return self.size

class FCQ(nn.Module):
    i=0
    def __init__(self,
        input_dim, # 입력 차원
        output_dim, # 출력 차원
        hidden_dims=(32,32), # 은닉계층
        activation_fc=F.relu): # 활성화 함수
        super(FCQ, self).__init__()
        self.activation_fc = activation_fc
        # 입력계층->은닉계층1
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        # 은닉계층
        self.hidden_layers = nn.ModuleList() # 신경망 모듈 리스트 초기화
        # 은닉계층-1 개 추가
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1]) # 은닉계층 추가
            self.hidden_layers.append(hidden_layer) # 모듈 리스트에 추가
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim) # 마지막 은닉계층->출력계층

        # CUDA(GPU) 가용하면 GPU로 디바이스 설정
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device) # 신경망 계산 디바이스 설정
        self.to(self.device) # 모듈의 파라미터/버퍼 등을 해당 디바이스로 이동(캐스트)
        # 입력이 파이토치 텐서가 아니면 텐서로 변환
    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor): # 파이토치 텐서가 아니면 텐서로 변환
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0) # 첫번째 차원 추가
        return x
        # 신경망 순전파 함수
    def forward(self, state):
        x = self._format(state) # 텐서로 변환
        x = self.activation_fc(self.input_layer(x)) # 입력계층
        self.i =0
        for hidden_layer in self.hidden_layers: # 은닉계층들
            self.i +=1
            x = self.activation_fc(hidden_layer(x))
            x = self.output_layer(x) # 출력계층
        return x

    # 넘파이 경험 튜플을 텐서로 변환, 추가된 메서드
    def load(self, experiences):
        states, actions, new_states, rewards, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)  # 상태
        actions = torch.from_numpy(actions).long().to(self.device)  # 행동
        new_states = torch.from_numpy(new_states).float().to(self.device)  # 다음 상태
        rewards = torch.from_numpy(rewards).float().to(self.device)  # 보상
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)  # 종료 상태 여부
        return states, actions, new_states, rewards, is_terminals
# 훈련 입실론 그리디 행동 선택 클래스, 기하급수적으로 감가되는 입실론
class EGreedyExpStrategy():

    def __init__(self, init_epsilon=1.0, min_epsilon=0.1, decay_steps=20000):
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.decay_steps = decay_steps
        self.min_epsilon = min_epsilon
        # 0.01/[10**-2, .......... 10**0] - 0.01
        self.epsilons = 0.01 / np.logspace(-2, 0, decay_steps, endpoint=False) - 0.01
        # init_epsilon와 min_epsilon 사이 값으로 조정
        self.epsilons = self.epsilons * (init_epsilon - min_epsilon) + min_epsilon
        self.t = 0 # 현재 스텝의 입실론 지정 인덱스
        self.exploratory_action_taken = None
        # 입실론 선택하고 인덱스 증가
    def _epsilon_update(self):
        # 미리 감가 계산된 입실론들 중 선택
        self.epsilon = self.min_epsilon if self.t >= self.decay_steps else self.epsilons[self.t]
        self.t += 1
        return self.epsilon
        # 감가된 입실론 그리디 전략으로 온라인 신경망 사용하여 행동 선택
    def select_action(self, model, state):
        self.exploratory_action_taken = False
        with torch.no_grad(): # 순전파 예측, 연산 기록 해제
            # 신경망 예측 Q 함수 값을 (기울기 연산 기록 해제 후) 넘파이로 변환 후 미니배치 차원 제거
            q_values = model(state).detach().cpu().data.numpy().squeeze()
        if np.random.rand() > self.epsilon: # 최대 Q-함수 행동 선택(그리디)
            action = np.argmax(q_values)
        else: # 입실론 확률로 탐색
            action = np.random.randint(len(q_values))
        self._epsilon_update() # 입실론 갱신(감가)
        self.exploratory_action_taken = action != np.argmax(q_values) # 탐색 여부 표시
        return action # 선택된 행동 리턴
class GreedyStrategy():
    def __init__(self):
        self.exploratory_action_taken = False
    def select_action(self, model, state):
        with torch.no_grad(): # 순전파 예측, 연산 기록 해제
        # 신경망 예측 Q 함수 값을 (기울기 연산 기록 해제 후) 넘파이로 변환 후 미니배치 차원 제거
            q_values = model(state).cpu().detach().data.numpy().squeeze()
        return np.argmax(q_values) # 최대 Q-함수 행동 선택(그리디)
## DQN 에이전트 클래스
class DQN(base_agent.BaseAgent):
    def __init__(self,
            replay_buffer_fn, # 재현 버퍼 람다함수
            value_model_fn, # 신경망 구성 람다 함수
            value_optimizer_fn, # 신경망 최적화기법 람다함수
            value_optimizer_lr, # 학습 속도
            training_strategy_fn, # 훈련 행동 선택 전략
            evaluation_strategy_fn, # 평가 행동 선택 전략
            n_warmup_batches, # 워밍업 미니배치 수, 샘플링 위한 버퍼의 최소 샘플 수 지정에 사용
            update_target_every_steps): # 타겟 신경망 갱신 스텝 수
        # 에이전트 클래스 속성 초기화
        super(DQN, self).__init__()
        self.replay_buffer_fn = replay_buffer_fn # 재현 버퍼 람다 함수
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.training_strategy_fn = training_strategy_fn
        self.evaluation_strategy_fn = evaluation_strategy_fn
        self.n_warmup_batches = n_warmup_batches # 워밍업 미니배치 수
        self.update_target_every_steps = update_target_every_steps # 타겟 신경망 갱신 스텝
        self.render = False
        self.mineral=0

        # 신경망 학습 모델: 타겟 예측(신경망 예측), 손실함수 계산, 역전파 수행(신경망 파라미터 갱신)
    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences
        batch_size = len(is_terminals)
        # max_a_q_sp = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)
        # 온라인 신경망으로 최대 행동-가치 함수 값의 행동 선택
        argmax_a_q_sp = self.online_model(next_states).max(1)[1]
        # 타겟 신경망으로 행동-가치함수 예측
        q_sp = self.target_model(next_states).detach()
        # 온라인 신경망의 행동으로 타겟의 최대 행동-가치함수 값 선택
        max_a_q_sp = q_sp[np.arange(batch_size), argmax_a_q_sp].unsqueeze(1)
        target_q_sa = rewards + (self.gamma * max_a_q_sp * (1 - is_terminals))  # 타겟 계산
        q_sa = self.online_model(states).gather(1, actions)  # 온라인 신경망 사용하여 현재 상태 Q-함수 예측
        td_error = q_sa - target_q_sa
        value_loss = td_error.pow(2).mul(0.5).mean()  # 손실함수 계산
        # 기울기 계산 (역전파)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
    # 입실론 그리디 전략 사용 행동 선택 후 수행하고 다음 상태 전이
    def interaction_step(self, state, env):
        # 행동 선택

        unit_info,_,_=self.Pretreatment(state)
        action = self.training_strategy.select_action(self.online_model, unit_info)
        # 행동 수행, 상태 전이
        _S = env.step(self.actionT(action,state))
        new_state, reward, is_terminal = self.Pretreatment(_S[0])
        experience = (unit_info, action, reward, new_state, float(is_terminal))
        self.replay_buffer.store(experience)  # 경험 튜플 재현 버퍼에 저장
        self.episode_reward[-1] += reward  # 보상 추가
        self.episode_timestep[-1] += 1  # 타임 스텝 증가
        self.episode_exploration[-1] += int(self.training_strategy.exploratory_action_taken)  # 탐색 횟수 증가
        return new_state, is_terminal, _S

    # 온라인 신경망 파라미터 사용하여 타겟 신경망 갱신
    def update_network(self):
        for target, online in zip(self.target_model.parameters(), self.online_model.parameters()):
            target.data.copy_(online.data)

    # 신경망 훈련
    def train(self, env, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward):
        training_start, last_debug_time = time.time(), float('-inf')
        # 초기화
        self.seed = seed
        self.gamma = gamma
        torch.manual_seed(self.seed);
        np.random.seed(self.seed);
        random.seed(self.seed)  # 랜덤 시드
        nS, nA = _NS ,_NA  # 상태/행동 수
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []
        self.episode_exploration = []
        # 타겟/온라인 신경망 객체 생성
        self.target_model = self.value_model_fn(nS, nA)
        self.online_model = self.value_model_fn(nS, nA)
        #self.target_model.load_state_dict(torch.load("./target_model"))
        #self.online_model.load_state_dict(torch.load("./online_model"))
        self.update_network()  # 타겟/온라인 신경망 초기화
        # 온라인 신경망의 최적화 기법, 재현 버퍼, 행동선택 전략 객체 생성
        self.value_optimizer = self.value_optimizer_fn(self.online_model, self.value_optimizer_lr)
        self.replay_buffer = self.replay_buffer_fn()
        self.training_strategy = self.training_strategy_fn()
        self.evaluation_strategy = self.evaluation_strategy_fn()
        # 에피소드 당 출력 값 저장 리스트 초기화
        result = np.empty((max_episodes, 5))
        result[:] = np.nan
        training_time = 0
        # 에피소드 반복
        for episode in range(1, max_episodes + 1):
            episode_start = time.time()  # 에피소드 시작 시간
            # 상태, 통계 변수 초기화
            state=env.reset()
            is_terminal= False
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)
            # 종료 상태 도달 시까지 반복

            while not is_terminal:
                # 행동 수행하고 상태 전이
                _, is_terminal ,state = self.interaction_step(state[0], env)
                # 샘플링을 위한 최소 재현 버퍼 크기 지정
                min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
                # 재현 버퍼에서 샘플링된 미니배치 데이터 사용하여 신경망 최적화
                if len(self.replay_buffer) > min_samples:
                    experiences = self.replay_buffer.sample()  # 재현 버퍼에서 샘플링
                    experiences = self.online_model.load(experiences)  # 넘파이 경험 튜플을 텐서로 변환
                    self.optimize_model(experiences)  # 신경망 최적화(역전파 수행하여 파라미터 갱신)
                # 지정된 타임 스텝(10) 마다 타겟 신경망 갱신
                if np.sum(self.episode_timestep) % self.update_target_every_steps == 0:
                    self.update_network()
            # 에피소드 당 통계 출력
            episode_elapsed = time.time() - episode_start  # 경과시간
            self.episode_seconds.append(episode_elapsed)  # 경과시간 리스트에 추가
            training_time += episode_elapsed  # 훈련시간
            total_step = int(np.sum(self.episode_timestep))  # 누적 타임 스텝
            # 한 에피소드 훈련 종료 후 그리디 행동 선택 적용하여 에피소드의 누적 보상 계산
            evaluation_score, _ = self.evaluate(self.online_model, env)
            self.evaluation_scores.append(evaluation_score)  # 누적 보상 리스트에 추가
            # 마지막 10/100 스텝 평균 훈련 보상/표준편차
            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])
            # 마지막 1100 스텝 평균 평가 보상/표준편차
            mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
            std_100_eval_score = np.std(self.evaluation_scores[-100:])
            # 마지막 100스텝 탐색 수/표준편차 출력
            lst_100_exp_rat = np.array(
                self.episode_exploration[-100:]) / np.array(self.episode_timestep[-100:])
            mean_100_exp_rat = np.mean(lst_100_exp_rat)
            std_100_exp_rat = np.std(lst_100_exp_rat)
            # 경과 시간 계산
            wallclock_elapsed = time.time() - training_start
            # 에피소드 당 결과 값 리스트에 저장
            result[episode - 1] = total_step, mean_100_reward, \
                                  mean_100_eval_score, training_time, wallclock_elapsed
            LEAVE_PRINT_EVERY_N_SECS = 20   # 행 출력 지속 시간
            ERASE_LINE = '\x1b[2K'  # 행 삭제
            reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS
            # 디버그 출력 지속시간 초과 플래그 설정
            reached_max_minutes = wallclock_elapsed >= max_minutes * 60  # 최대 제한 시간 초과 플래그 설정
            reached_max_episodes = episode >= max_episodes  # 최대 제한 에피소드 수 초과 플래그 설정
            reached_goal_mean_reward = mean_100_eval_score >= goal_mean_100_reward
            # 에피소드 마지막 100스텝 최대 보상 초과 플래그 설정
            # 훈련 중지 플래그 설정: 최대 수행 시간이나 에피소드 초과 시 또는 마지막 100 스텝 평균 최대 보상 도달 시
            training_is_over = reached_max_minutes or reached_max_episodes or reached_goal_mean_reward
            # 에피소드 출력 메시지: 경과시간, 에피소드 번호, 스텝 수, 마지막 10/100 스텝 평균 훈련 보상/표준편차,
            # 마지막 100 스텝 평균 탐색 수/표준편차, 마지막 100 스텝 평균 평가 보상/표준편차
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
            debug_message = 'el {}, ep {:04}, ts {:06}, '
            debug_message += 'ar 10 {:05.1f}\u00B1{:05.1f}, '
            debug_message += '100 {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'ex 100 {:02.1f}\u00B1{:02.1f}, '
            debug_message += 'ev {:05.1f}\u00B1{:05.1f}'
            debug_message = debug_message.format(
                elapsed_str, episode - 1, total_step, mean_10_reward, std_10_reward,
                mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat,
                mean_100_eval_score, std_100_eval_score)
            print(debug_message, end='\r', flush=True)

            # 디버그 출력 시간 초과 또는 훈련 종료 시 출력 메시지 화면 출력
            if reached_debug_time or training_is_over:
                print(ERASE_LINE + debug_message, flush=True)
                last_debug_time = time.time()
            if training_is_over:  # 훈련 종료 시 출력
                if reached_max_minutes: print(u'--> reached_max_minutes \u2715')
                if reached_max_episodes: print(u'--> reached_max_episodes \u2715')
                if reached_goal_mean_reward: print(u'--> reached_goal_mean_reward \u2713')
                break
            # 훈련 종료 후 모델 평가 수행
        final_eval_score, score_std = self.evaluate(self.online_model, env, n_episodes=100)
        wallclock_time = time.time() - training_start  # 총 훈련 시간(모델 평가 포함)
        print('Training complete.')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(final_eval_score, score_std, training_time, wallclock_time))
        env.close();
        del env
        return result, final_eval_score, training_time, wallclock_time

    # 에피소드/훈련 종료 후 그리디 전략으로 보상 계산
    def evaluate(self, eval_policy_model, eval_env, n_episodes=1):
        rs = []
        for _ in range(n_episodes):
            _S =eval_env.reset()
            s,_,_= self.Pretreatment(_S[0])
            done = False
            rs.append(0)
            self.j = 0
            while not done:
                self.j +=1
                a = self.evaluation_strategy.select_action(eval_policy_model, s)
            # 그리디 전략으로 행동 수행하고 상태 전이
                _S = eval_env.step(self.actionT(a, _S[0]))
                s, r, done= self.Pretreatment(_S[0])
                rs[-1] += r  # 보상 추가
        print(rs[-1])
        return np.mean(rs), np.std(rs)
    def Pretreatment(self,obs):
        units_info = obs.observation.feature_units
        units_info = [[unit.x, unit.y, unit.alliance] for unit in units_info]#원하는 정보만 뽑아서 새로 생성
        units_info =np.array(units_info)#리스트 넘파이 변환
        units_info= units_info.flatten() #평탄화
        info= np.zeros(shape=(_NS+2)) #feature_units의 길이는 매번 변화함 길이를 맞춰줄 필요성 있음 최대 갯수로 0 초기화
        units_info = sum(info ,units_info)#0으로 초기화한 배열에 units_info더함
        units_info[-2]=len(obs.observation.feature_units)#뒤에서 2번째 인덱스에 유닛 갯수 지정
        if obs.observation.player.minerals==self.mineral:#먹지 못했을경우
            reward = -1
        else:
            reward = obs.observation.player.minerals - self.mineral#먹었을경우
        if np.isnan(obs.observation.last_actions):
            units_info[-1] = 1000 #마지막으로 아무행동도 안했을경우 마지막 값이 1000이 된다.
            reward= -10
        units_info = 2. * (units_info - np.min(units_info)) / np.ptp(units_info) - 1#정규화
        self.mineral =   obs.observation.player.minerals
        done= obs.last() #시간이 다되면
        if done: self.mineral =0

        return units_info ,reward ,done
    def actionT(self,a,obs):
        action = []
        if len(obs.observation.single_select)==0:#선택된 유닛이 존재하지 않으면
            for unit in obs.observation.feature_units:
                if unit.unit_type == 48:
                    action.append( actions.FUNCTIONS.select_point("select", (unit.x, unit.y)))#선택을 한다.
                    break
        if len(obs.observation.feature_units) < a :#지금 없는 유닛을 선택할 경우
            action = [actions.FUNCTIONS.no_op()]#아무것도 하지 않는다
        elif MOVE_SCREEN in obs.observation.available_actions:#있는 유닛을 선택했고 이동이 가능한 경우
            action.append( actions.FunctionCall(MOVE_SCREEN, [NOT_QUEUED, [obs.observation.feature_units[a-1][12],
                                                                           obs.observation.feature_units[a-1][13]]]))
        return action

def sum(num1, num2):
    for i in range(len(num2)):
        num1[i] += num2[i]
    return num1
def main(args):
    ## DQN 메인 루틴
    dqn_results = []
    # 각기 다른 시드 값으로 수행
    #SEEDS = (12,24,62,15,23)
    SEEDS = (12,)
    for seed in SEEDS:
        # 환경 세팅 파라미터
        environment_settings = {
            'gamma': 0.88, # 감가율(할인율)
            'max_minutes': 3600, # 최대 수행 시간
            'max_episodes': 100, # 최대 에피소드 수
            'goal_mean_100_reward': 5000 # 마지막 100 스텝 평균 최대 보상
        }
        # 신경망 모델 구성 람다 함수
        value_model_fn = lambda nS, nA: FCQ(nS+2, nA, hidden_dims=(512,128))
        # 신경망 최적화 기법 지정 람다 함수
        value_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
        value_optimizer_lr = 0.001  # 학습 속도
        # 훈련 입실론 그리디 탐색 지정 람다 함수
        training_strategy_fn = lambda: EGreedyExpStrategy(init_epsilon=1.0, min_epsilon=0.3, decay_steps=20000)
        # 평가 입실론 그리디 탐색 지정 람다 함수
        evaluation_strategy_fn = lambda: GreedyStrategy()
        # 재현 버퍼 지정 람다 함수
        replay_buffer_fn = lambda: ReplayBuffer(max_size=50000, batch_size=64)
        n_warmup_batches = 5  # 워밍업 배치 크기
        update_target_every_steps = 10  # 타겟 신경망 갱신 스텝 수
        # 환경 파라미터 지정 및 환경 생성
        gamma, max_minutes, max_episodes, goal_mean_100_reward = environment_settings.values()
        with sc2_env.SC2Env(map_name=MAPNAME, players=players, \
                            agent_interface_format=interface, \
                            step_mul=APM, game_steps_per_episode=UNLIMIT, \
                            visualize=VISUALIZE, realtime=REALTIME) as env:

            # DQN 에이전트 생성
            agent = DQN(replay_buffer_fn, value_model_fn, value_optimizer_fn, value_optimizer_lr, training_strategy_fn,
                        evaluation_strategy_fn, n_warmup_batches, update_target_every_steps)
            agent.setup(env.observation_spec(), env.action_spec())
            # DQN 신경망 훈련
            result, final_eval_score, training_time, wallclock_time = agent.train(env, seed, gamma, max_minutes,
                                                                                  max_episodes, goal_mean_100_reward)
            print("---------------------")
            dqn_results.append(result)  # 시드 결과 리스트에 추가
    # 시드 결과 리스트를 넘파이 배열로 변환
    dqn_results = np.array(dqn_results)
    torch.save(agent.online_model.state_dict(), "./online_model")
    torch.save(agent.target_model.state_dict(), "./target_model")
    # 그래픽용 자료 추출
    dqn_max_t, dqn_max_r, dqn_max_s, dqn_max_sec, dqn_max_rt = np.max(dqn_results, axis=0).T
    dqn_min_t, dqn_min_r, dqn_min_s, dqn_min_sec, dqn_min_rt = np.min(dqn_results, axis=0).T
    dqn_mean_t, dqn_mean_r, dqn_mean_s, dqn_mean_sec, dqn_mean_rt = np.mean(dqn_results, axis=0).T
    dqn_x =500 # DQN 결과와 비교를 위해 스케일 맞춤
    # 에피소드 당 (마지막 100스탭) 평균 보상(훈련,평가), 총 스텝 수, 훈련/경과 시간 그래픽
    import matplotlib.pyplot as plt
    import matplotlib.pylab as pylab
    plt.style.use('fivethirtyeight')
    params = {
        'figure.figsize': (15, 8),
        'font.size': 24,
        'legend.fontsize': 20,
        'axes.titlesize': 28,
        'axes.labelsize': 24,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20
    }
    pylab.rcParams.update(params)
    # 서브그래프 분할
    fig, axs = plt.subplots(2, 1, figsize=(15,30), sharey=False, sharex=True)
    # 마지막 100 스텝 훈련 시 평균 보상
    axs[0].plot(dqn_max_r, 'y', linewidth=1)
    axs[0].plot(dqn_min_r, 'y', linewidth=1)
    axs[0].plot(dqn_mean_r, 'y', label='DQN', linewidth=2)
    axs[0].fill_between(dqn_x, dqn_min_r, dqn_max_r, facecolor='y', alpha=0.3)
    # 마지막 100 스텝 평가 시 평균 보상
    axs[1].plot(dqn_max_s, 'y', linewidth=1)
    axs[1].plot(dqn_min_s, 'y', linewidth=1)
    axs[1].plot(dqn_mean_s, 'y', label='DQN', linewidth=2)
    axs[1].fill_between(dqn_x, dqn_min_s, dqn_max_s, facecolor='y', alpha=0.3)

    # 차트 제목
    axs[0].set_title('default (Training)')
    axs[1].set_title('default (Evaluation)')
    plt.xlabel('Episodes')
    axs[0].legend(loc='upper left')
    plt.show()
app.run(main)