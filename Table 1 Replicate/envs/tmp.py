from hopper_env import HopperEnv
import matplotlib.pyplot as plt
env = HopperEnv(data_types=['size0.03'], seed=0)
s = env.reset()
for _ in range(100):
    a = env.action_space.sample()
    s, _, _, _ = env.step(a)
ob = env.render('rgb_array')
ob = ob[125:475, 70:425,:]
plt.imshow(ob)
plt.axis('off')
plt.show()
plt.savefig('hopper_size0.03.pdf', bbox_inches='tight')

#from walker2d_env import Walker2dEnv
#env = Walker2dEnv(data_types=['size0.05'], seed=0)
#s = env.reset()
#ob = env.render('rgb_array')
#plt.imshow(ob)
#plt.show()
#plt.savefig('walker_size0.05.png')
