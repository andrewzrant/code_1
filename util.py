#从曝光记录中寻找指定广告某天的曝光量
def find_history(ad_id,date):
    tvalue = data_expos[(data_expos['ad_id'] == ad_id) & (data_expos['ad_req_date'] == date)]['expos']
    if tvalue.shape[0] == 0:
        return 0
    else:
        return tvalue.values[0]
		
#先默认，如果修改后第二天未修改，则提取第二天的曝光量，否则，一直到修改结束
def find_op(input_ad):
    ad_id = input_ad['ad_id'].values[0]
    tdata = data_op_ad[data_op_ad['ad_id'] == ad_id]
    if tdata.shape[0] == 0: #广告只创建过，未在op记录中存在
        return -1
    if (tdata['op_type']==2).sum() == 0: #广告虽然定向，但是未新建相关设置，直接去除
        return -1
    tdata.sort_values(by='op_time',inplace=True)
    tdata.reset_index(drop=True,inplace=True)
    #记录新建设置
    ttdata = tdata[tdata['op_type']==2]
    new_idx = ttdata.index
    for idx in new_idx:
        if ttdata['op_item'].loc[idx] == 2:
            input_ad['bid'] = tdata['op_item_value'].loc[idx]
        elif ttdata['op_item'].loc[idx] == 3:
            input_ad['pep_groups'] = tdata['op_item_value'].loc[idx]
        elif ttdata['op_item'].loc[idx] == 4:
            input_ad['ad_time'] = tdata['op_item_value'].loc[idx]
    input_ad['ad_sta'] = 0
    expose_data = -1
    #记录更改项，如果第二天没有更改，则统计第二天的曝光量        
    idx_list = tdata.index.drop(new_idx)
    for inx,idx in enumerate(idx_list):
        #记录下操作的内容
        if tdata['op_item'].loc[idx] == 1:
            input_ad['ad_sta'] = tdata['op_item_value'].loc[idx]
        elif tdata['op_item'].loc[idx] == 2:
            input_ad['bid'] = tdata['op_item_value'].loc[idx]
        elif tdata['op_item'].loc[idx] == 3:
            input_ad['pep_groups'] = tdata['op_item_value'].loc[idx]
        elif tdata['op_item'].loc[idx] == 4:
            input_ad['ad_time'] = tdata['op_item_value'].loc[idx]  
            
        if inx<idx_list.shape[0]-1:
            #判断第二天是否有修改记录，如果有，则不统计曝光量，否则统计第二天的曝光量
            idx_1 = idx_list[inx+1]
            now_date = tdata['op_date'].loc[idx]
            next_date = tdata['op_date'].loc[idx_1]
            if (next_date-now_date).days>1:#说明第二天未修改
                if int(input_ad['ad_sta'].values[0]) == 1:#广告有效
                    date = now_date +datetime.timedelta(days=1)
                    input_ad['expos_time'] = date
                    input_ad['expos'] = find_history(ad_id,date)
                    try:
                        expose_data = expose_data.append(input_ad)
                    except:
                        expose_data = input_ad
        else:#最后一条记录
            if int(input_ad['ad_sta'].values[0]) == 1:#广告有效
                now_date = tdata['op_date'].loc[idx]
                date = now_date +datetime.timedelta(days=1)
                input_ad['expos_time'] = date
                input_ad['expos'] = find_history(ad_id,date)
                try:
                    expose_data = expose_data.append(input_ad)
                except:
                    expose_data = input_ad
    return expose_data
#生成训练集	
idx_list = data_static_ad['ad_id'].values
op_idx_list = data_op_ad['ad_id'].values
for idx in tqdm(idx_list):
    if idx not in op_idx_list:
        continue
    input_ad = data_static_ad[data_static_ad['ad_id'] == idx]
    try:
        getexpos = find_op(input_ad)
    except:
        print(f"id:{idx} is err")
        break
    if type(getexpos) is int:
        continue
    else:
        try:
            traindata = traindata.append(getexpos)
        except:
            traindata = getexpos