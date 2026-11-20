import sys
from os import path
sys.path.insert(0, path.abspath(path.join(path.dirname(__file__), '..')))
from Src.parser import Parser
from Src.Algorithms.DSPO_plus_SPO import DSPO_plus_SPO

def make_obj(warmup=5, rampup=10, max_w=0.7, data_len=1, episode=0, first_data=0):
    obj = DSPO_plus_SPO.__new__(DSPO_plus_SPO)
    obj.spo_training_data = {i: None for i in range(data_len)}
    obj.spo_warmup_episodes = warmup
    obj.spo_rampup_episodes = rampup
    obj.max_spo_loss_weight = max_w
    obj.spo_current_episode_count = episode
    obj.spo_first_data_episode = first_data
    return obj

def main():
    a = Parser().get_parser().parse_args([])
    assert (a.spo_loss_weight, a.spo_warmup_episodes, a.spo_rampup_episodes,
            a.spo_batch_size, a.spo_buffer_size, a.spo_label_sample_size) == (0.7, 5, 10, 32, 0, 0)
    # warmup 内为 0
    assert DSPO_plus_SPO._get_current_spo_weight(make_obj(episode=4)) == 0.0
    # rampup 中点线性
    w = DSPO_plus_SPO._get_current_spo_weight(make_obj(episode=10))
    assert abs(w - 0.7 * 0.5) < 1e-9, w
    # 封顶
    assert abs(DSPO_plus_SPO._get_current_spo_weight(make_obj(episode=100)) - 0.7) < 1e-9
    # 无数据恒 0
    assert DSPO_plus_SPO._get_current_spo_weight(make_obj(data_len=0, episode=100)) == 0.0
    # 调度以首个有标签episode为锚点:112个初始阶段episode之后才开始计warmup/rampup
    # effective=120-112=8,超过warmup(5),rampup进度=(8-5)/10=0.3
    w2 = DSPO_plus_SPO._get_current_spo_weight(make_obj(episode=120, first_data=112))
    assert abs(w2 - 0.7 * 0.3) < 1e-9, w2
    print("OK")

if __name__ == "__main__":
    main()
