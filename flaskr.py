import os
from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash, jsonify
import sys


app = Flask(__name__)
app.config.from_object(__name__)


@app.route('/')
def load():
	return render_template('index.html')
