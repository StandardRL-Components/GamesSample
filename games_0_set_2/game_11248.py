import gymnasium as gym
import os
import pygame
import numpy as np
from gymnasium.spaces import MultiDiscrete
import pygame.gfxdraw
import math
import random
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Engage in rhythmic combat in a neon-drenched arena. Time your attacks and blocks "
        "to the beat to defeat waves of enemies."
    )
    user_guide = (
        "Use arrow keys (↑↓←→) to move. Press space to attack and shift to block. "
        "Time your actions with the rhythm bar for bonuses."
    )
    auto_advance = True

    # --- Constants ---
    # Colors (Neon Cyberpunk Theme)
    COLOR_BG = (15, 15, 25)
    COLOR_GRID = (40, 40, 60)
    COLOR_PLAYER = (0, 255, 255)
    COLOR_PLAYER_GLOW = (0, 150, 150)
    COLOR_ENEMY = (255, 0, 100)
    COLOR_ENEMY_GLOW = (150, 0, 60)
    COLOR_ENEMY_ATTACK = (255, 50, 50)
    COLOR_SWORD_SLASH = (200, 255, 255)
    COLOR_BLOCK_SHIELD = (100, 200, 255, 100)  # RGBA for transparency
    COLOR_UI_TEXT = (220, 220, 255)
    COLOR_STAMINA_BAR = (0, 255, 100)
    COLOR_STAMINA_EMPTY = (100, 100, 120)
    COLOR_RHYTHM_BAR = (255, 255, 0)
    COLOR_RHYTHM_HIT = (0, 255, 0)
    COLOR_RHYTHM_MISS = (255, 0, 0)

    # Screen Dimensions
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400

    # Game Parameters
    MAX_STEPS = 5000
    PLAYER_SPEED = 4.0
    PLAYER_RADIUS = 12
    MAX_STAMINA = 100.0
    STAMINA_REGEN_RATE = 0.25
    STAMINA_COST_ATTACK = 20
    STAMINA_COST_BLOCK = 30
    STAMINA_COST_MOVE = 0.1
    ATTACK_DURATION = 8  # frames
    ATTACK_RADIUS = 50
    BLOCK_DURATION = 15  # frames
    RHYTHM_PERIOD = 90  # frames (3 seconds at 30fps)
    RHYTHM_HIT_WINDOW = 8  # frames (+/- 4 frames around the beat)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 32, bold=True)

        # State variables are initialized in reset()
        self.player_pos = None
        self.player_stamina = None
        self.player_action_timer = None
        self.player_action = None  # "attack", "block", None
        self.is_exhausted = None

        self.enemies = None
        self.particles = None
        self.floating_texts = None

        self.wave = None
        self.rhythm_timer = None
        self.last_rhythm_feedback = None  # Stores color and timer for feedback

        self.prev_space_held = False
        self.prev_shift_held = False

        self.steps = 0
        self.score = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0

        self.player_pos = pygame.Vector2(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2)
        self.player_stamina = self.MAX_STAMINA
        self.player_action = None
        self.player_action_timer = 0
        self.is_exhausted = False

        self.enemies = []
        self.particles = []
        self.floating_texts = []

        self.wave = 0  # Will be incremented to 1 by _spawn_wave
        self._spawn_wave()

        self.rhythm_timer = 0
        self.last_rhythm_feedback = {"color": self.COLOR_RHYTHM_BAR, "timer": 0}

        self.prev_space_held = False
        self.prev_shift_held = False

        return self._get_observation(), self._get_info()

    def step(self, action):
        # Unpack factorized action
        movement_action = action[0]
        space_held = action[1] == 1
        shift_held = action[2] == 1

        # Detect button presses (rising edge)
        space_pressed = space_held and not self.prev_space_held
        shift_pressed = shift_held and not self.prev_shift_held
        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        reward = 0

        # --- Update Game Logic ---
        self._update_rhythm()
        reward += self._handle_player_actions(movement_action, space_pressed, shift_pressed)
        self._update_player_state()
        reward += self._update_enemies()
        self._update_particles()
        self._update_floating_texts()

        # Check for wave completion
        if not self.enemies:
            reward += 100  # Wave survival reward
            self.score += 1000
            self._spawn_wave()
            # Sound: wave_complete.wav
            self._add_floating_text(f"WAVE {self.wave}", self.player_pos, self.COLOR_RHYTHM_HIT, 60)

        self.score += reward
        self.steps += 1

        terminated = self.player_stamina <= 0 or self.steps >= self.MAX_STEPS
        if self.player_stamina <= 0:
            reward -= 100  # Terminal penalty for exhaustion
        
        truncated = self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            float(reward),
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_player_actions(self, movement_action, space_pressed, shift_pressed):
        reward = 0
        can_act = self.player_action is None and not self.is_exhausted

        # Movement
        move_vec = pygame.Vector2(0, 0)
        if movement_action == 1: move_vec.y = -1
        elif movement_action == 2: move_vec.y = 1
        elif movement_action == 3: move_vec.x = -1
        elif movement_action == 4: move_vec.x = 1

        if move_vec.length() > 0:
            move_vec.normalize_ip()
            self.player_pos += move_vec * self.PLAYER_SPEED
            self.player_stamina -= self.STAMINA_COST_MOVE
            # Clamp player position
            self.player_pos.x = max(self.PLAYER_RADIUS, min(self.player_pos.x, self.SCREEN_WIDTH - self.PLAYER_RADIUS))
            self.player_pos.y = max(self.PLAYER_RADIUS, min(self.player_pos.y, self.SCREEN_HEIGHT - self.PLAYER_RADIUS))

        # Rhythmic Actions
        is_on_beat = abs(self.rhythm_timer - self.RHYTHM_PERIOD) < self.RHYTHM_HIT_WINDOW or abs(self.rhythm_timer) < self.RHYTHM_HIT_WINDOW

        # Attack (Spacebar)
        if space_pressed and can_act and self.player_stamina >= self.STAMINA_COST_ATTACK:
            self.player_action = "attack"
            self.player_action_timer = self.ATTACK_DURATION
            self.player_stamina -= self.STAMINA_COST_ATTACK

            if is_on_beat:
                reward += 0.1
                self.last_rhythm_feedback = {"color": self.COLOR_RHYTHM_HIT, "timer": 15}
            else:
                reward -= 0.1
                self.last_rhythm_feedback = {"color": self.COLOR_RHYTHM_MISS, "timer": 15}

            # Check for hits
            for enemy in self.enemies:
                if self.player_pos.distance_to(enemy["pos"]) < self.ATTACK_RADIUS + enemy["radius"]:
                    enemy["health"] -= 25 * (1.5 if is_on_beat else 1)
                    reward += 1.0
                    self._add_particles(20, enemy["pos"], self.COLOR_PLAYER)
                    self._add_floating_text("-25", enemy["pos"] + pygame.Vector2(0, -20), self.COLOR_UI_TEXT)
                    if enemy["health"] <= 0:
                        reward += 5.0
                        self.score += 50
                        self._add_particles(50, enemy["pos"], self.COLOR_ENEMY, 5.0)

        # Block (Shift)
        elif shift_pressed and can_act and self.player_stamina >= self.STAMINA_COST_BLOCK:
            self.player_action = "block"
            self.player_action_timer = self.BLOCK_DURATION
            self.player_stamina -= self.STAMINA_COST_BLOCK

            if is_on_beat:
                reward += 0.1
                self.last_rhythm_feedback = {"color": self.COLOR_RHYTHM_HIT, "timer": 15}
            else:
                reward -= 0.1
                self.last_rhythm_feedback = {"color": self.COLOR_RHYTHM_MISS, "timer": 15}

        return reward

    def _update_player_state(self):
        # Action timer
        if self.player_action_timer > 0:
            self.player_action_timer -= 1
            if self.player_action_timer == 0:
                self.player_action = None

        # Stamina management
        if not self.player_action:
            self.player_stamina = min(self.MAX_STAMINA, self.player_stamina + self.STAMINA_REGEN_RATE)

        if self.is_exhausted:
            if self.player_stamina > self.MAX_STAMINA / 4:
                self.is_exhausted = False
        elif self.player_stamina <= 0:
            self.player_stamina = 0
            self.is_exhausted = True

    def _update_enemies(self):
        reward = 0
        enemies_to_remove = []
        for enemy in self.enemies:
            if enemy["health"] <= 0:
                enemies_to_remove.append(enemy)
                continue

            # Movement AI
            direction = self.player_pos - enemy["pos"]
            if direction.length() > 0:
                enemy["pos"] += direction.normalize() * enemy["speed"]

            # Attack AI
            enemy["attack_timer"] -= 1
            if enemy["attack_timer"] <= 0:
                # Reset timer
                enemy["attack_timer"] = enemy["attack_cooldown"]

                # Check if player is hit
                if self.player_pos.distance_to(enemy["pos"]) < enemy["attack_radius"]:
                    if self.player_action == "block":
                        # Successful block
                        self._add_particles(30, self.player_pos, self.COLOR_BLOCK_SHIELD)
                        self._add_floating_text("BLOCKED!", self.player_pos, self.COLOR_RHYTHM_HIT)
                    else:
                        # Player gets hit
                        damage = 10 * (2 if self.is_exhausted else 1)
                        self.player_stamina -= damage
                        reward -= 1.0
                        self._add_particles(30, self.player_pos, self.COLOR_ENEMY_ATTACK)
                        self._add_floating_text(f"-{damage}", self.player_pos, self.COLOR_ENEMY_ATTACK)

        self.enemies = [e for e in self.enemies if e not in enemies_to_remove]
        return reward

    def _update_rhythm(self):
        self.rhythm_timer = (self.rhythm_timer + 1) % (self.RHYTHM_PERIOD + 1)
        if self.last_rhythm_feedback["timer"] > 0:
            self.last_rhythm_feedback["timer"] -= 1

    def _spawn_wave(self):
        self.wave += 1
        num_enemies = 1 + self.wave // 2

        for _ in range(num_enemies):
            angle = random.uniform(0, 2 * math.pi)
            dist = self.SCREEN_WIDTH / 2.2
            pos = pygame.Vector2(
                self.SCREEN_WIDTH / 2 + math.cos(angle) * dist,
                self.SCREEN_HEIGHT / 2 + math.sin(angle) * dist
            )

            health = 100 * (1.05 ** (self.wave - 1))
            speed = 0.8 * (1.02 ** (self.wave - 1))
            attack_cooldown = max(60, 180 - self.wave * 5)

            self.enemies.append({
                "pos": pos,
                "radius": 10,
                "health": health,
                "max_health": health,
                "speed": speed,
                "attack_timer": random.randint(0, int(attack_cooldown)),
                "attack_cooldown": attack_cooldown,
                "attack_radius": 40
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_enemies()
        self._render_player()
        self._render_particles()
        self._render_ui()
        self._render_floating_texts()

        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave}

    def _render_background(self):
        for i in range(0, self.SCREEN_WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.SCREEN_HEIGHT), 1)
        for i in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.SCREEN_WIDTH, i), 1)

    def _render_player(self):
        pos = (int(self.player_pos.x), int(self.player_pos.y))

        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS + 4, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS + 4, self.COLOR_PLAYER_GLOW)
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)

        if self.player_action == "attack":
            progress = self.player_action_timer / self.ATTACK_DURATION
            angle_offset = -math.pi / 2
            start_angle = angle_offset - (1 - progress) * math.pi
            end_angle = angle_offset + (1 - progress) * math.pi
            rect = pygame.Rect(pos[0] - self.ATTACK_RADIUS, pos[1] - self.ATTACK_RADIUS, self.ATTACK_RADIUS * 2, self.ATTACK_RADIUS * 2)
            pygame.draw.arc(self.screen, self.COLOR_SWORD_SLASH, rect, start_angle, end_angle, 3)

        if self.player_action == "block":
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            radius = int(self.PLAYER_RADIUS * 2.5)
            alpha = 100 - (self.player_action_timer / self.BLOCK_DURATION) * 80
            pygame.gfxdraw.filled_circle(s, pos[0], pos[1], radius, (*self.COLOR_BLOCK_SHIELD[:3], int(alpha)))
            pygame.gfxdraw.aacircle(s, pos[0], pos[1], radius, (*self.COLOR_BLOCK_SHIELD[:3], int(alpha)))
            self.screen.blit(s, (0, 0))

    def _render_enemies(self):
        for enemy in self.enemies:
            pos = (int(enemy["pos"].x), int(enemy["pos"].y))
            radius = int(enemy["radius"])

            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius + 3, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius + 3, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_ENEMY)

            health_pct = max(0, enemy["health"] / enemy["max_health"])
            bar_width = radius * 2
            bar_height = 4
            bar_pos = (pos[0] - radius, pos[1] - radius - 10)
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_GLOW, (*bar_pos, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_RHYTHM_HIT, (*bar_pos, bar_width * health_pct, bar_height))

            if enemy["attack_timer"] < 30:
                progress = (30 - enemy["attack_timer"]) / 30
                current_radius = int(enemy["attack_radius"] * progress)
                alpha = int(150 * (1 - progress))
                if alpha > 0:
                    pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], current_radius, (*self.COLOR_ENEMY_ATTACK, alpha))

    def _render_ui(self):
        bar_width = 200
        bar_height = 20
        stamina_pct = self.player_stamina / self.MAX_STAMINA
        pygame.draw.rect(self.screen, self.COLOR_STAMINA_EMPTY, (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_STAMINA_BAR, (10, 10, bar_width * stamina_pct, bar_height))
        if self.is_exhausted:
            exhausted_text = self.font_small.render("EXHAUSTED", True, self.COLOR_ENEMY_ATTACK)
            self.screen.blit(exhausted_text, (15, 12))

        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        wave_text = self.font_small.render(f"WAVE: {self.wave}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH - wave_text.get_width() - 10, 30))

        bar_y = self.SCREEN_HEIGHT - 30
        bar_width = 300
        bar_height = 10
        bar_x = (self.SCREEN_WIDTH - bar_width) / 2

        pygame.draw.rect(self.screen, self.COLOR_GRID, (bar_x, bar_y, bar_width, bar_height), border_radius=5)

        hit_zone_width = (self.RHYTHM_HIT_WINDOW / self.RHYTHM_PERIOD) * bar_width
        hit_zone_x = bar_x + bar_width - hit_zone_width / 2
        pygame.draw.rect(self.screen, self.COLOR_RHYTHM_BAR, (hit_zone_x, bar_y - 5, hit_zone_width, bar_height + 10), 2, border_radius=5)

        beat_progress = self.rhythm_timer / self.RHYTHM_PERIOD
        beat_x = bar_x + beat_progress * bar_width

        feedback_color = self.last_rhythm_feedback["color"] if self.last_rhythm_feedback["timer"] > 0 else self.COLOR_RHYTHM_BAR
        pygame.draw.line(self.screen, feedback_color, (beat_x, bar_y - 5), (beat_x, bar_y + bar_height + 5), 3)

    def _render_particles(self):
        for p in self.particles:
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'].x), int(p['pos'].y), int(p['radius']), p['color'])

    def _update_particles(self):
        particles_to_remove = []
        for p in self.particles:
            p['pos'] += p['vel']
            p['lifespan'] -= 1
            p['radius'] *= 0.95
            if p['lifespan'] <= 0 or p['radius'] < 0.5:
                particles_to_remove.append(p)
        self.particles = [p for p in self.particles if p not in particles_to_remove]

    def _add_particles(self, count, pos, color, speed_mult=1.0):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1.0, 3.0) * speed_mult
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'color': color,
                'radius': random.uniform(2, 5),
                'lifespan': random.randint(10, 20)
            })

    def _render_floating_texts(self):
        for ft in self.floating_texts:
            text_surf = self.font_small.render(ft['text'], True, ft['color'])
            text_surf.set_alpha(ft['alpha'])
            self.screen.blit(text_surf, (ft['pos'].x - text_surf.get_width() / 2, ft['pos'].y))

    def _update_floating_texts(self):
        texts_to_remove = []
        for ft in self.floating_texts:
            ft['pos'].y -= 0.5
            ft['lifespan'] -= 1
            ft['alpha'] = max(0, int(255 * (ft['lifespan'] / ft['max_lifespan'])))
            if ft['lifespan'] <= 0:
                texts_to_remove.append(ft)
        self.floating_texts = [ft for ft in self.floating_texts if ft not in texts_to_remove]

    def _add_floating_text(self, text, pos, color, lifespan=30):
        self.floating_texts.append({
            'text': text,
            'pos': pos.copy(),
            'color': color,
            'lifespan': lifespan,
            'max_lifespan': lifespan,
            'alpha': 255
        })

    def close(self):
        pygame.font.quit()
        pygame.quit()


if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # Make sure to unset the dummy video driver if you want to see the game
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]
        
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Cyber Blade Dancer")
    clock = pygame.time.Clock()

    running = True
    total_reward = 0

    while running:
        movement = 0  # 0: none
        space = 0  # 0: released
        shift = 0  # 0: released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            movement = 4

        if keys[pygame.K_SPACE]:
            space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift = 1

        action = [movement, space, shift]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Total Reward: {total_reward:.2f}, Wave: {info['wave']}")
            total_reward = 0
            obs, info = env.reset()

        clock.tick(30)  # Run at 30 FPS

    env.close()