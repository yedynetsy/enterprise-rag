import os

import requests
import streamlit as st


API_URL = os.getenv(
    "RAG_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")


st.set_page_config(
    page_title="Enterprise RAG",
    page_icon="📄",
    layout="centered",
)

st.title("📄 Ask your document")
st.caption("Answers grounded in ust_manual.pdf, with sources.")

with st.form("question_form"):
    question = st.text_area(
        "Your question",
        placeholder="What materials can be used to absorb a fuel spill?",
        height=100,
    )

    submitted = st.form_submit_button(
        "Get answer",
        type="primary",
    )


if submitted:
    st.session_state.pop("last_result", None)
    question = question.strip()

    if not question:
        st.warning("Please enter a question.")
        st.stop()

    try:
        with st.spinner("Searching the document and preparing an answer…"):
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question},
                timeout=(5, 90),
            )

            response.raise_for_status()
            result = response.json()

    except requests.exceptions.Timeout:
        st.error("The request timed out. Please try again later.")
        st.stop()

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the API. "
            "Make sure the server is running."
        )
        st.stop()

    except requests.exceptions.HTTPError:
        st.error(
            f"The service returned HTTP error {response.status_code}."
        )
        st.stop()

    except requests.exceptions.RequestException:
        st.error("Could not send the request to the service.")
        st.stop()

    except ValueError:
        st.error("The service returned an invalid response.")
        st.stop()

    st.session_state["last_result"] = {
        "question": question,
        "result": result,
    }


if "last_result" in st.session_state:
    saved = st.session_state["last_result"]
    result = saved["result"]

    st.divider()
    st.caption(f"Question: {saved['question']}")
    st.subheader("Answer")

    if result["status"] == "insufficient_context":
        st.info(
            "The retrieved passages do not contain enough information."
        )

    st.markdown(result["answer"])

    if result["sources"]:
        st.subheader("Sources")

        for source in result["sources"]:
            label = (
                f"Source {source['source_id']} · "
                f"{source['document']} · "
                f"page {source['page']}"
            )

            with st.expander(label):
                st.write(source["text"])
                st.caption(
                    f"Similarity score: {source['score']:.3f}"
                )
