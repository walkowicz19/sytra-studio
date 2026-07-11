import './app.css'
import 'bootstrap-icons/font/bootstrap-icons.css'
import { mount } from 'svelte'
import App from './App.svelte'
import { initTheme } from './store.svelte'

initTheme()
mount(App, { target: document.getElementById('app')! })
