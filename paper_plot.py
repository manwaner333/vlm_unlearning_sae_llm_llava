import matplotlib.pyplot as plt
import numpy as np

# # 示例数据：四组，每组三条数据
# group_labels = ['Forget Set', 'Test Set', 'Retain Set', 'Real Celebrity']
# bar_labels = ['Intervene in Vision Encoder', 'Intervene in LLMs', 'Intervention on Vision Encoder & LLMs']  # 每组中三条数据的标签
# ### 5%
# # data = [
# #     [44.55, 36.38, 33.24],   
# #     [40.12, 33.35, 32.79],   
# #     [42.98, 42.88, 42.22],   
# #     [50.31, 49.33, 48.48]    
# # ]

# # data = [
# #     [0.57, 0.49, 0.42],   
# #     [0.33, 0.29, 0.18],   
# #     [0.59, 0.54, 0.52],   
# #     [0.45, 0.44, 0.42]    
# # ]

# # data = [
# #     [14.66, 12.98, 7.88],   
# #     [20.24, 18.70, 16.23],   
# #     [18.06, 17.07, 16.85],   
# #     [13.41, 14.11, 12.94]    
# # ]

# ### 10%
# # data = [
# #     [41.14, 34.8, 30.25],   
# #     [39.93, 36.31, 31.85],   
# #     [44.05, 43.38, 42.24],   
# #     [39.93, 36.31, 31.85]    
# # ]

# # data = [
# #     [0.57, 0.49, 0.43],   
# #     [0.44, 0.32, 0.28],   
# #     [0.62, 0.61, 0.59],   
# #     [0.45, 0.44, 0.45]    
# # ]

# # data = [
# #     [20.02, 15.77, 11.88],   
# #     [19.58, 15.77, 11.24],   
# #     [19.78, 18.28, 16.98],   
# #     [14.86, 13.22, 12.44]    
# # ]

# ### 15%
# # data = [
# #     [41.65, 35.24, 30.33],   
# #     [41.48, 33.95, 30.70],   
# #     [44.23, 41.54, 43.55],   
# #     [50.84, 46.82, 45.82]    
# # ]
# # data = [
# #     [0.53, 0.48, 0.41],   
# #     [0.44, 0.34, 0.30],   
# #     [0.59, 0.59, 0.57],   
# #     [0.45, 0.40, 0.40]    
# # ]

# data = [
#     [18.51, 15.19, 12.00],   
#     [19.88, 15.85, 11.23],   
#     [22.46, 21.25, 21.10],   
#     [13.99, 10.97, 10.25]    
# ]


# # 转置数据，使其变为3行4列，便于分组画图
# data = np.array(data).T  # shape: (3, 4)

# x = np.arange(len(group_labels))  # [0, 1, 2, 3]
# bar_width = 0.20

# fig, ax = plt.subplots(figsize=(8, 6))

# # 绘制三组条形，每组向右移动一定距离
# for i in range(len(bar_labels)):
#     ax.bar(x + i * bar_width, data[i], width=bar_width, label=bar_labels[i])

# # 设置坐标轴和标题
# ax.set_xlabel('Datasets')
# # ax.set_ylabel('Rouge')
# ax.set_ylabel('Classification (%)')
# # ax.set_title('')
# ax.set_xticks(x + bar_width)
# ax.set_xticklabels(group_labels)
# ax.legend()

# plt.tight_layout()
# # plt.savefig('cloze_task.png', dpi=300)
# # plt.savefig('rouge_task.png', dpi=300)
# # plt.savefig('classification_task.png', dpi=300)
# # plt.savefig('classification_task_10.png', dpi=300)
# # plt.savefig('rouge_task_10.png', dpi=300)
# # plt.savefig('cloze_task_10.png', dpi=300)
# # plt.savefig('classification_task_15.png', dpi=300)
# # plt.savefig('rouge_task_15.png', dpi=300)
# plt.savefig('cloze_task_15.png', dpi=300)
# # plt.show()




# 示例数据：四组，每组三条数据
### 5%
# data = [
#     [44.55, 36.38, 33.24],   
#     [40.12, 33.35, 32.79],   
#     [42.98, 42.88, 42.22],   
#     [50.31, 49.33, 48.48]    
# ]

# data = [
#     [0.57, 0.49, 0.42],   
#     [0.33, 0.29, 0.18],   
#     [0.59, 0.54, 0.52],   
#     [0.45, 0.44, 0.42]    
# ]

# data = [
#     [14.66, 12.98, 7.88],   
#     [20.24, 18.70, 16.23],   
#     [18.06, 17.07, 16.85],   
#     [13.41, 14.11, 12.94]    
# ]

### 10%
# data = [
#     [41.14, 34.8, 30.25],   
#     [39.93, 36.31, 31.85],   
#     [44.05, 43.38, 42.24],   
#     [39.93, 36.31, 31.85]    
# ]

# data = [
#     [0.57, 0.49, 0.43],   
#     [0.44, 0.32, 0.28],   
#     [0.62, 0.61, 0.59],   
#     [0.45, 0.44, 0.45]    
# ]

# data = [
#     [20.02, 15.77, 11.88],   
#     [19.58, 15.77, 11.24],   
#     [19.78, 18.28, 16.98],   
#     [14.86, 13.22, 12.44]    
# ]

### 15%
# data = [
#     [41.65, 35.24, 30.33],   
#     [41.48, 33.95, 30.70],   
#     [44.23, 41.54, 43.55],   
#     [50.84, 46.82, 45.82]    
# ]
# data = [
#     [0.53, 0.48, 0.41],   
#     [0.44, 0.34, 0.30],   
#     [0.59, 0.59, 0.57],   
#     [0.45, 0.40, 0.40]    
# ]

# data = [
#     [18.51, 15.19, 12.00],   
#     [19.88, 15.85, 11.23],   
#     [22.46, 21.25, 21.10],   
#     [13.99, 10.97, 10.25]    
# ]

# data = [
#     [39.20, 27.28],
#     [8.16, 7.60],   
# ]

# data = [
#     [46.92, 37.52],
#     [14.87, 18.83],   
# ]

# data = [
#     [39.30, 26.28],
#     [19.03, 13.43],   
# ]

# data = [
#     [40.34, 56.62],
#     [13.07, 12.81],   
# ]

# data = [
#     [36.40, 24.10],
#     [14.96, 8.80],   
# ]

# data = [
#     [47.35, 37.13],
#     [17.64, 16.26],   
# ]

# data = [
#     [35.36, 28.34],
#     [14.03, 8.45],   
# ]

# data = [
#     [38.91, 52.89],
#     [15.32, 9.56],   
# ]

# data = [
#     [37.34, 23.32],
#     [10.84, 13.16],   
# ]

# data = [
#     [47.05, 40.05],
#     [22.56, 19.64],   
# ]

# data = [
#     [33.67, 27.73],
#     [14.02, 8.44],   
# ]

# data = [
#     [40.16, 53.48],
#     [14.37, 7.56],   
# ]


# group_labels = ['Classification', 'Cloze']
# bar_labels = ['Text Question', 'Image Text Question']
group_labels = ['Image+Text', 'Pure Text']
# bar_labels = ['One SAE', 'Two SAEs']
bar_labels = ['SAE@L', 'SAE@V+L']

# data = [
#     [12.00, 20.72],
#     [4.42, 22.02],   
# ]

# classification_forget
# data = [
#     [4.42, 12.00,],
#     [22.02, 20.72,],   
# ]

# classification_test
# data = [
#     [3.56, 6.30,],
#     [15.34, 13.72,],   
# ]

# classification_retain
# data = [
#     [4.56, 2.36,],
#     [2.72, 2.23,],   
# ]

# real celebrity
# data = [
#     [4.11, 3.00,],
#     [4.82, 4.82,],   
# ]

# rouge_forget
# data = [
#     [0.07, 0.18,],
#     [0.29, 0.26,],   
# ]

# rouge_test
# data = [
#     [0.02, 0.21,],
#     [0.23, 0.25,],   
# ]

# rouge_retain
# data = [
#     [0.16, 0.09,],
#     [0.16, 0.12,],   
# ]

# rouge_real_celebrity
data = [
    [0.013, 0.013,],
    [0.044, 0.054,],   
]

# cloze_forget
# data = [
#     [2.95, 12.84,],
#     [12.47, 12.40,],   
# ]

# cloze_test
# data = [
#     [3.87, 8.97,],
#     [6.74, 6.57,],   
# ]

# cloze_retain
# data = [
#     [6.21, 5.07,],
#     [2.43, 2.01,],   
# ]

# cloze_real_celebrity
# data = [
#     [3.95, 3.1,],
#     [1.71, 1.9,],   
# ]

# 转置数据，使其变为3行4列，便于分组画图
data = np.array(data).T  # shape: (3, 4)

x = np.arange(len(group_labels))  # [0, 1, 2, 3]
bar_width = 0.2

fig, ax = plt.subplots(figsize=(4, 4))


# 绘制三组条形，每组向右移动一定距离
for i in range(len(bar_labels)):
    ax.bar(x + i * bar_width, data[i], width=bar_width, label=bar_labels[i])

# 设置坐标轴和标题
fontsize = 16
# ax.set_xlabel('Datasets')
# ax.set_ylabel('Rouge')
# ax.set_ylabel('ACC Difference (%)', fontsize=fontsize)
ax.set_ylabel('Rouge Difference', fontsize=14)
# ax.set_title('Forget Set', fontsize=fontsize)
# ax.set_title('Test Set', fontsize=fontsize)
# ax.set_title('Retain Set', fontsize=fontsize)
ax.set_title('Real Celebrity', fontsize=fontsize)
ax.set_xticks(x + bar_width)
ax.set_xticklabels(group_labels, fontsize=fontsize)
ax.tick_params(axis='y', labelsize=fontsize)
# ax.legend(fontsize=fontsize)
# ax.legend(ncol=2, loc='upper center', fontsize=fontsize)
# ax.legend(loc='upper center', bbox_to_anchor=(0.4, 1.4), ncol=2, fontsize=fontsize)


plt.tight_layout()
# plt.savefig('cloze_task.png', dpi=300)
# plt.savefig('rouge_task.png', dpi=300)
# plt.savefig('classification_task.png', dpi=300)
# plt.savefig('classification_task_10.png', dpi=300)
# plt.savefig('rouge_task_10.png', dpi=300)
# plt.savefig('cloze_task_10.png', dpi=300)
# plt.savefig('classification_task_15.png', dpi=300)
# plt.savefig('rouge_task_15.png', dpi=300)
# plt.savefig('cloze_task_15.png', dpi=300)
# plt.savefig('mutiple_questions_forget.png', dpi=300)
# plt.savefig('mutiple_questions_retain.png', dpi=300)
# plt.savefig('mutiple_questions_test.png', dpi=300)
# plt.savefig('mutiple_questions_celebrity.png', dpi=300)
# plt.savefig('mutiple_questions_forget_10.png', dpi=300)
# plt.savefig('mutiple_questions_retain_10.png', dpi=300)
# plt.savefig('mutiple_questions_test_10.png', dpi=300)
# plt.savefig('mutiple_questions_celebrity_10.png', dpi=300)
# plt.savefig('mutiple_questions_forget_15.png', dpi=300)
# plt.savefig('mutiple_questions_retain_15.png', dpi=300)
# plt.savefig('mutiple_questions_test_15.png', dpi=300)
# plt.savefig('mutiple_questions_celebrity_15.png', dpi=300)
# plt.savefig('figure/classification_forget.png', dpi=300)
# plt.savefig('figure/classification_test.png', dpi=300)
# plt.savefig('figure/classification_retain.png', dpi=300)
# plt.savefig('figure/classification_real_celebrity.png', dpi=300)
# plt.savefig('figure/rouge_forget.png', dpi=300)
# plt.savefig('figure/rouge_test.png', dpi=300)
# plt.savefig('figure/rouge_retain.png', dpi=300)
# plt.savefig('figure/rouge_real_celebrity.png', dpi=300)
# plt.savefig('figure/cloze_forget.png', dpi=300)
# plt.savefig('figure/cloze_test.png', dpi=300)
# plt.savefig('figure/cloze_retain.png', dpi=300)
# plt.savefig('figure/cloze_real_celebrity.png', dpi=300)
plt.subplots_adjust(left=0.2)
plt.savefig('figure/rouge_real_celebrity.png', dpi=300)
# plt.show()
