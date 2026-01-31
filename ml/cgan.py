class SeismicCGan:
    def __init__(self, noise_dim=None, img_size=128, n_classes=6, learning_rate=0.0002, beta_1=0.5):
        self.img_size = img_size
        self.n_classes = n_classes
        self.lambda_l1 = 100.0

        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.d_optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1)
        self.g_optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1)

        self.g_model = self.build_generator()
        self.d_model = self.build_discriminator()

    def generator_loss(self, disc_generated_logits, gen_output, target):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_logits), disc_generated_logits)
        l1_loss  = tf.reduce_mean(tf.abs(target - gen_output))
        total    = gan_loss + self.lambda_l1 * l1_loss
        return total, gan_loss, l1_loss

    def discriminator_loss(self, disc_real_logits, disc_fake_logits):
        real_loss = self.loss_object(tf.ones_like(disc_real_logits), disc_real_logits)
        fake_loss = self.loss_object(tf.zeros_like(disc_fake_logits), disc_fake_logits)
        total = real_loss + fake_loss
        return total, real_loss, fake_loss

    def _fft_mag(self, x):
        # x: (B,H,W,1) float32
        x2 = tf.squeeze(x, axis=-1)  # (B,H,W)
        X  = tf.signal.fft2d(tf.cast(x2, tf.complex64))
        return tf.abs(X)             # (B,H,W)

    def texture_fft_l1(self, x_true, x_fake, scales=(1, 2, 4)):
        loss = 0.0
        for s in scales:
            if s > 1:
                xt = tf.nn.avg_pool2d(x_true, ksize=s, strides=s, padding="SAME")
                xf = tf.nn.avg_pool2d(x_fake, ksize=s, strides=s, padding="SAME")
            else:
                xt, xf = x_true, x_fake

            mt = self._fft_mag(xt)
            mf = self._fft_mag(xf)
            loss += tf.reduce_mean(tf.abs(mt - mf))
        return loss / float(len(scales))

    def amp_log_moments(self, x_true, x_fake, eps=1e-3):
        # a = log(|x| + eps)
        at = tf.math.log(tf.abs(x_true) + eps)
        af = tf.math.log(tf.abs(x_fake) + eps)

        axes = [1, 2, 3]  # H,W,C
        mt = tf.reduce_mean(at, axis=axes)
        mf = tf.reduce_mean(af, axis=axes)
        st = tf.math.reduce_std(at, axis=axes)
        sf = tf.math.reduce_std(af, axis=axes)

        # kurtosis
        ct = at - tf.reduce_mean(at, axis=axes, keepdims=True)
        cf = af - tf.reduce_mean(af, axis=axes, keepdims=True)
        kt = tf.reduce_mean(tf.pow(ct / (tf.math.reduce_std(at, axis=axes, keepdims=True) + 1e-6), 4), axis=axes)
        kf = tf.reduce_mean(tf.pow(cf / (tf.math.reduce_std(af, axis=axes, keepdims=True) + 1e-6), 4), axis=axes)

        d_mean = tf.reduce_mean(tf.abs(mt - mf))
        d_std  = tf.reduce_mean(tf.abs(st - sf))
        d_kurt = tf.reduce_mean(tf.abs(kt - kf))
        return d_mean, d_std, d_kurt

    def amp_quantiles(self, x_true, x_fake, qs=(0.5, 0.9, 0.95, 0.99)):
        # сравнение квантилей по |x|
        at = tf.reshape(tf.abs(x_true), [tf.shape(x_true)[0], -1])  # (B, HW)
        af = tf.reshape(tf.abs(x_fake), [tf.shape(x_fake)[0], -1])

        at_sorted = tf.sort(at, axis=-1)
        af_sorted = tf.sort(af, axis=-1)

        n = tf.cast(tf.shape(at_sorted)[-1], tf.float32)
        diffs = []

        for q in qs:
            idx = tf.cast(tf.round(q * (n - 1.0)), tf.int32)
            qt = tf.gather(at_sorted, idx, axis=-1)  # (B,)
            qf = tf.gather(af_sorted, idx, axis=-1)
            diffs.append(tf.reduce_mean(tf.abs(qt - qf)))

        return diffs  # list of scalars: [dq50, dq90, dq95, dq99]


    
    def build_generator(self):
        def encoder_block(layer_in, n_filters):
            init = tf.random_normal_initializer(0., 0.02)
    
            # downsample
            g = Conv2D(n_filters, 5, strides=2, padding='same', kernel_initializer=init)(layer_in)
            g = LeakyReLU(alpha=0.2)(g)
    
            # refine (stride=1)
            g = Conv2D(n_filters, 3, strides=1, padding="same", kernel_initializer=init)(g)
            g = LeakyReLU(alpha=0.2)(g)
            return g
    
        def decoder_block(layer_in, skip_in, n_filters, dropout=True):
            init = tf.random_normal_initializer(0., 0.02)
    
            # upsample
            g = Conv2DTranspose(n_filters, 4, strides=2, padding='same', kernel_initializer=init)(layer_in)
            if dropout:
                g = Dropout(0.2)(g)
    
            g = Activation('relu')(g)
            g = Concatenate()([g, skip_in])
    
            # refine (stride=1)
            g = Conv2D(n_filters, 3, strides=1, padding="same", kernel_initializer=init)(g)
            g = Activation("relu")(g)
            return g
    
        inputs = Input(shape=(self.img_size, self.img_size, self.n_classes))
    
        # encoder
        e1 = encoder_block(inputs, 64)
        e2 = encoder_block(e1, 128)
        e3 = encoder_block(e2, 256)
        e4 = encoder_block(e3, 512)
    
        # decoder
        d1 = decoder_block(e4, e3, 256, dropout=True)
        d2 = decoder_block(d1, e2, 128, dropout=False)
        d3 = decoder_block(d2, e1, 64, dropout=False)
    
        out_image = Conv2DTranspose(
            1, 4, strides=2, padding='same',
            kernel_initializer=tf.random_normal_initializer(0., 0.02),
            activation='tanh'
        )(d3)
    
        return Model(inputs, out_image, name="Generator")


    def build_discriminator(self):
        init = tf.random_normal_initializer(0., 0.02)
    
        in_label = Input((self.img_size, self.img_size, self.n_classes), name="input_label")
        in_image = Input((self.img_size, self.img_size, 1), name="input_image")
        x = Concatenate()([in_image, in_label])  # (H,W,1+C)
    
        def conv(x, f, s, bn=True):
            x = Conv2D(f, 5, strides=s, padding="same", kernel_initializer=init, use_bias=not bn)(x)
            if bn:
                x = BatchNormalization()(x)
            x = LeakyReLU(0.2)(x)
            #x = tf.keras.layers.Activation("sigmoid")(x)
            return x
    
        x = conv(x, 64, 2, bn=False)   # 128 -> 64
        x = conv(x, 128, 2, bn=True)   # 64 -> 32
        x = conv(x, 256, 2, bn=True)   # 32 -> 16
        x = conv(x, 512, 1, bn=True)   # 16 -> 16
        x = conv(x, 512, 1, bn=True)
    
        out = Conv2D(1, 4, strides=1, padding="same",
                     kernel_initializer=init)(x)
    
        return Model([in_image, in_label], out, name="Discriminator")


    # =========================
    # Train step with metrics
    # =========================
    @tf.function
    def train_step(self, image_batch, label_batch):
        # image_batch: (B,H,W,1)
        # label_batch: (B,H,W,C)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.g_model(label_batch, training=True)

            disc_real = self.d_model([image_batch, label_batch], training=True)
            disc_fake = self.d_model([tf.stop_gradient(generated_images), label_batch], training=True)
            disc_fake_for_g = self.d_model([generated_images, label_batch], training=True)

            gen_total, gen_gan, gen_l1 = self.generator_loss(disc_fake_for_g, generated_images, image_batch)
            disc_total, disc_real_loss, disc_fake_loss = self.discriminator_loss(disc_real, disc_fake)



            # ---- метрики качества (НЕ входят в оптимизацию) ----
            tex = self.texture_fft_l1(image_batch, generated_images, scales=(1, 2, 4))          # текстуры
            d_mean, d_std, d_kurt = self.amp_log_moments(image_batch, generated_images)         # ампл. моменты
            dq50, dq90, dq95, dq99 = self.amp_quantiles(image_batch, generated_images)          # хвосты

            # ---- диагностические метрики D ----
            # аккуратности по patch-карте
            real_p = tf.nn.sigmoid(disc_real)
            fake_p = tf.nn.sigmoid(disc_fake)
            d_real_acc = tf.reduce_mean(tf.cast(real_p > 0.5, tf.float32))
            d_fake_acc = tf.reduce_mean(tf.cast(fake_p < 0.5, tf.float32))

        gen_gradients = gen_tape.gradient(gen_total, self.g_model.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_total, self.d_model.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gen_gradients, self.g_model.trainable_variables))
        self.d_optimizer.apply_gradients(zip(disc_gradients, self.d_model.trainable_variables))

        # возвращаем всё нужное для логов
        return {
            "d_total": disc_total,
            "d_real_loss": disc_real_loss,
            "d_fake_loss": disc_fake_loss,
            "d_real_acc": d_real_acc,
            "d_fake_acc": d_fake_acc,
            "g_total": gen_total,
            "g_gan": gen_gan,
            "g_l1": gen_l1,
            "tex_fft": tex,
            "amp_d_mean": d_mean,
            "amp_d_std": d_std,
            "amp_d_kurt": d_kurt,
            "dq50": dq50,
            "dq90": dq90,
            "dq95": dq95,
            "dq99": dq99,
        }    
    
    def train(self, dataset, epochs=50):
        # history под новый train_step (который возвращает dict)
        history = {
            "d_total": [],
            "d_real_loss": [],
            "d_fake_loss": [],
            "d_real_acc": [],
            "d_fake_acc": [],
            "g_total": [],
            "g_gan": [],
            "g_l1": [],
            "tex_fft": [],
            "amp_d_mean": [],
            "amp_d_std": [],
            "amp_d_kurt": [],
            "dq50": [],
            "dq90": [],
            "dq95": [],
            "dq99": [],
        }
    
        for epoch in range(epochs):
            start = time.time()
    
            # собираем значения по батчам
            sums = {k: [] for k in history.keys()}
    
            for image_batch, label_batch in tqdm(dataset, desc=f"Epoch {epoch+1}/{epochs}"):
                logs = self.train_step(image_batch, label_batch)
    
                # logs — dict тензоров; приводим к float
                for k in sums.keys():
                    sums[k].append(float(logs[k]))
    
            # усредняем по эпохе
            for k in history.keys():
                history[k].append(float(np.mean(sums[k])))
    
            print(f"\nEpoch: {epoch+1}/{epochs}")
            print(f"  D total: {history['d_total'][-1]:.4f} | real: {history['d_real_loss'][-1]:.4f} | fake: {history['d_fake_loss'][-1]:.4f}")
            print(f"  D acc : real {history['d_real_acc'][-1]:.3f} | fake {history['d_fake_acc'][-1]:.3f}")
            print(f"  G total: {history['g_total'][-1]:.4f} | gan: {history['g_gan'][-1]:.4f} | l1: {history['g_l1'][-1]:.4f}")
            print(f"  Texture FFT: {history['tex_fft'][-1]:.4f}")
            print(f"  Amp: Δstd(log|x|)={history['amp_d_std'][-1]:.4f} | Δq95={history['dq95'][-1]:.4f} | Δq99={history['dq99'][-1]:.4f}")
            print(f"  Time for epoch: {time.time()-start:.2f} sec\n")
    
        print("\nPlotting training history...")
        self.plot_training_history(history)
        return history