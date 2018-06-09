# 2018 PKU Data Warehouse

#### 1、对XML文件的预处理：HandleXML.ipynb

**包括**：

——（1）将其中的类似'&ouml ;'的字符串删掉，这是unicode遗留问题

——（2）提取出1998-2007年间的（会议、作者）信息，写入到新文件 data/data.txt。每行的格式为：会议$作者1;作者2;.....作者n;