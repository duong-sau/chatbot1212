// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom';

json = {
    response: {
        request_hash: "mã yêu cầu (sha1)",
        answer:{
            left_answer:[
                    {
                        label: "nhãn",
                        answer: "link đến tài liệu",
                        first: "câu dẫn nhập",
                        highlight: "đoạn liên quan nhất"
                    }
                    ],
            right_answer:[
                    {
                        label: "nhãn",
                        answer: "link đến tài liệu",
                        first: "câu dẫn nhập",
                        highlight: "đoạn liên quan nhất"
                    }
                    ],
        }
    }
}
question = {
    params: {
        question:how can i analysis bootstrap,
        top_k: 5,
        left_method: T5,
        right_method: cosine,
        left_parameter: 2,
        right_parameter:BM25,
    }
}