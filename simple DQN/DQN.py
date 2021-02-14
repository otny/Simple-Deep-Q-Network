
# DQN

import numpy as np
import matplotlib.pyplot as plt
import gym

from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display

ENV = 'CartPole-v0'     # 課題名
GAMMA = 0.99            # 時間割引率
MAX_STEPS = 200         # 1試行のstep数
NUM_EPISODES = 500      # 最大試行回数
BATCH_SIZE = 32         # バッチサイズ
CAPACITY = 10000        # メモリサイズのキャパ


# 状態, 行動, 次の状態, 報酬　これらを一つにまとめる遷移(Transition)を作成
from collections import namedtuple  # タプルを使うための宣言
"""
# namedtupleの使い方
from collections import namedtuple

Tr = namedtuple('tr', ('name_a', 'value_b'))
Tr_object = Tr('名前Aです', 100)

print(Tr_object)
print(Tr_object.name_a, Tr_object.value_b)
"""
Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))




# 動画保存用
def display_frame_as_gif(frames):
    """
    Displays as list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[0])
        
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)

    anim.save('movie_carpole_DQN.mp4')  # 動画のファイル名と保存
    display(display_animation(anim, default_mode='loop'))







class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY    # メモリの最大長さ
        self.memory = []            # 経験を保存する変数
        self.index = 0              # 保存するindexを示す変数

    def push(self, state, action, next_state, reward):
        """
        transition = ('state', 'action', 'next_state', 'reward')をメモリに保存する
        簡単に言うと経験をためる
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)    # メモリが満タンでないときは足す

        # namedtupleのTransitionを使用し, 値とフィールド名をペアにして保存
        self.memory[self.index] = Transition(state, action, next_state, reward)
        self.index = (self.index + 1) % self.capacity   # 保存するindexを1つずらす

    def sample(self, batch_size):
        """
        batch_size分だけ, ランダムに保存内容を取り出す
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        関数lenに対して, 現在の変数memoryの長さを返す
        """
        return len(self.memory)








# エージェントが持つ脳となるクラス, DQNの実行
# Q関数をディープラーニングのネットワークをクラスとして定義

import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions  # CartPoleの行動数2(左右)を取得

        # 経験を記憶するメモリオブジェクトを生成
        self.memory = ReplayMemory(CAPACITY)

        # ニューラルネットワークを構築
        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(num_states, 32))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(32, 32))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(32, num_actions))

        # ネットワークの形を出力
        print(self.model)

        # 最適化手法の設定
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def replay(self):
        """
        Experience Replayでネットワークの結合パラメータを学習
        """
        # ----------------------
        # 1. メモリサイズの確認
        # ----------------------
        # 1.1 メモリサイズがミニバッチより小さい間は何もしない
        if len(self.memory) < BATCH_SIZE:
            return

        # ----------------------
        # 2. ミニバッチの作成
        # ----------------------
        # 2.1 メモリからミニバッチ分のデータを取り出す.
        transitions = self.memory.sample(BATCH_SIZE)

        # 2.2 各変数をミニバッチに対応する形に変形
        # transitionは1stepごとの(state, action, state_next, reward)がBATCH_SIZEぶん格納されている
        # つまり, (state, action, state_next, reward)*BATCH_SIZE をミニバッチにしたい. つまり
        # (state*BATCH_SIZE, action*BATCH_SIZE, state_next*BATCH_SIZE, reward*BATCH_SIZE) にするんご
        batch = Transition(*zip(*transitions))

        # 2.3 各変数の要素をミニバッチに対応する形に変形する
        # 例えば state の場合, [torch.FloatTensor of size 1*4] が BATCH_SIZEぶん並んでいるのですが, 
        # それを torch.FloatTensor of size BATCH_SIZE*4 に変換します.
        # catはConecatenates（結合）のこと
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # ----------------------
        # 2. 教師信号となるQ(s_t, a_t)値を求める
        # ----------------------
        # 3.1 ネットワークを推論モードに切り替える
        self.model.eval()
