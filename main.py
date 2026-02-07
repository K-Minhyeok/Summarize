from flask import Flask, request, jsonify
from crawl.crawler import get_contents
from summarize.summarizer import summarize_content

app = Flask(__name__)

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    limit = data.get("limit")
    summary = []

    # 크롤링
    articles = get_contents(limit)

    # 요약
    for article in articles:
        if article["content"] != "Fail to scrap":
            summary.append(summarize_content(article["content"]))
        else :
            summary.append("Fail to scrap")

    for i in range(len(summary)):
        print(f'{i+1}: {summary[i]}')
        print("------------------------------")
        
    # Parsing 
    # ex) 요약 내용: ~~~ --- 
    #     장점 : ~~~  ---
    #     단점 : ~~~ ---




    # return jsonify({
    #     "status": "success",
    #     "title" : title,
    #     "summary": summary
    # })
    return "hihihihi"

if __name__ == "__main__":
    app.run(debug=True)
