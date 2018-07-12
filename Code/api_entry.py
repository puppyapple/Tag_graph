import flask
import json
import Code.recommendation
from flask import request

server = flask.Flask(__name__)
@server.route("/recommendation", methods=["post"])
def rec():
    comp_name = request.form.to_dict().get("comp_name")
    comp_info = request.form.to_dict().get("comp_info")
    response_num = int(request.form.to_dict().get("response_num"))
    data = []
    code = -1
    # print(comp_name)
    # print(type(response_num))
    # test = Code.recommendation.multi_process_rank(comp_name="安徽秒秒科技有限责任公司", response_num=1).to_dict(orient="row")
    # print(test)
    try:
        data = Code.recommendation.multi_process_rank(comp_name=comp_name, comp_info=comp_info, response_num=response_num).to_dict(orient="row")
        # print(data)
    except:
        print("请求失败！")
    else:
        code = 200
    finally:
        result_json = json.dumps({"code": code, "data": data})
        return result_json

server.run(port=8999,debug=True,host="172.30.212.219")