## Generate lunar permitivity map using pre-trained inversion model

---

- Setup a `venv` using `requirements.txt`
- Pre-trained model is available in `model` directory
- Download the radar backsactter images and preferably put them inside `data` directory
- Edit the `conf.json` file acoordingly
- Finally run the prediction as folloWs (`debug` part is optional and can be omitted):

```bash
python predict.py -c conf.json -d debug.json
```

- Works best with available `cuda`hardware