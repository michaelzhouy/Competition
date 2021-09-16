## OTTO solutions
1. magic特征
```python
data['magic_rank'] = data.groupby(['date'])['article_id'].rank()
```
2. 历史销量的统计值
3. 当天, 全局count特征, 当天/全局
4. 当日, 当周, 全局orders_1h、orders_2h统计特征
5. 当日, 当周, 全局收藏、点赞、评论统计特征
6. 商品价格, 降价幅度统计特征
7. 当日, 当周, 全局nunique统计特征