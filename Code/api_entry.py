import flask
import json
from Code import recommendation
from flask import request

server = flask.Flask(__name__)
@server.route("/recommendation", methods=["post"])
def rec():
    comp_name = request.form.to_dict().get("comp_name")
    comp_info = request.form.to_dict().get("comp_info")
    metric = request.form.to_dict().get("metric")
    response_num = int(request.form.to_dict().get("response_num"))
    print(comp_name)
    print(comp_info)
    print(metric)
    print(response_num)
    data = []
    code = -1

    try:
        data = recommendation.sparse_cal(comp_name=comp_name, metric=metric, comp_info=comp_info, response_num=response_num)[2].to_dict(orient="row")
        print(data)
    except:
        print("请求失败！")
    else:
        code = 200
    finally:
        result_json = json.dumps({"code": code, "data": data})
        return result_json

server.run(port=8999,debug=True,host="172.30.212.219")