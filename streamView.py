import streamlit as st

st.info("This is a testing app  ")

if "page" not in st.session_state:
  st.session_state.page = 1

if "info" not in st.session_state:
  st.session_state.info = {}

def go_to_page1(name):
  if name:
    st.session_state.info["name"] = name
    st.session_state.page = 2

def go_to_page2():
  st.session_state.page = 3

if st.session_state.page == 1:

  st.title("Welcome")
  
  st.write("Enter your info")
  name = st.text_input(label ="Name", value = st.session_state.info.get("name",""))

  button = st.button("Submit", on_click=go_to_page1, args=(name,))
  
  if button and not name:
    st.write("Please fill in your info")

  else:
    st.write("") 

  st.write(st.session_state.page)

  

elif st.session_state.page == 2:
  st.write(f"**Hello** {st.session_state.info.get('name', '')}!")
  st.write("How may I help you today?")
  st.text_input("")
  st.button("Exit", on_click=go_to_page2)

elif st.session_state.page == 3:
  st.header("Goodbye...")
  st.write(st.session_state.info.get('name', ''))

else:
  st.write("")

st.sidebar.header('Specify Input Parameter')

with st.expander('**Info**'):
  st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla facilisi. Sed vehicula sapien at dui pretium, in blandit lectus malesuada. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Curabitur nec lorem eget urna bibendum congue. Integer posuere nibh ut tellus faucibus, vel accumsan odio facilisis. Suspendisse potenti. Phasellus dignissim leo vel justo fermentum, a porttitor turpis bibendum. Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium.")
