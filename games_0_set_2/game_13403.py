import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T21:21:01.942207
# Source Brief: brief_03403.md
# Brief Index: 3403
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Engage in tactical space combat. Charge your laser, manage your heat signature with camouflage, and destroy waves of enemies."
    )
    user_guide = (
        "Controls: Use arrow keys to aim. Hold space to charge your laser and release to fire. Hold shift to activate camouflage."
    )
    auto_advance = True

    # --- Game Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (10, 5, 25)
    COLOR_STAR_NEAR = (200, 200, 255)
    COLOR_STAR_MID = (150, 150, 200)
    COLOR_STAR_FAR = (100, 100, 150)
    
    COLOR_PLAYER = (0, 255, 150)
    COLOR_PLAYER_GLOW = (0, 255, 150, 30)
    
    COLOR_ENEMY = (255, 50, 50)
    COLOR_ENEMY_GLOW = (255, 50, 50, 40)

    COLOR_PLAYER_LASER = (100, 200, 255)
    COLOR_ENEMY_LASER = (255, 100, 50)
    
    COLOR_HEAT_SIG = (255, 150, 0)
    COLOR_CAMO_SHIMMER = (200, 150, 255)

    COLOR_UI_TEXT = (220, 220, 240)
    COLOR_UI_HEALTH = (0, 200, 100)
    COLOR_UI_HEAT = (255, 120, 0)
    COLOR_UI_CHARGE = (100, 200, 255)
    
    # Player
    PLAYER_MAX_HEALTH = 100
    PLAYER_AIM_SPEED = 0.05
    PLAYER_MAX_HEAT = 100
    PLAYER_HEAT_COOLDOWN = 0.4
    PLAYER_CAMO_HEAT_COOLDOWN_BOOST = 1.2
    PLAYER_FIRE_COOLDOWN = 5 # steps
    
    # Laser
    LASER_CHARGE_RATE = 2.5
    LASER_MIN_CHARGE_TO_FIRE = 10
    LASER_SPEED = 15
    
    # Camouflage
    CAMO_DETECTION_THRESHOLD = 30.0

    # Enemy
    ENEMY_BASE_HEALTH = 30
    ENEMY_BASE_FIRE_RATE = 90 # steps
    ENEMY_SPEED = 1.0
    
    # Episode
    MAX_STEPS = 5000

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
        
        self.stars = self._generate_stars(200)

        self.player_pos = None
        self.player_health = None
        self.player_heat = None
        self.player_angle = None
        self.laser_charge = None
        self.is_camouflaged = None
        self.player_fire_cooldown_timer = None
        self.was_space_held = None
        
        self.enemies = []
        self.player_bullets = []
        self.enemy_bullets = []
        self.particles = []

        self.wave = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.last_heat_above_threshold = None
        
        # This reset call is needed to initialize the state, but the return values are not used here.
        # self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = np.array([self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2], dtype=np.float32)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_heat = 0.0
        self.player_angle = -math.pi / 2
        self.laser_charge = 0.0
        self.is_camouflaged = False
        self.player_fire_cooldown_timer = 0
        self.was_space_held = False

        self.enemies = []
        self.player_bullets = []
        self.enemy_bullets = []
        self.particles = []

        self.wave = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_heat_above_threshold = False
        
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        self.steps += 1
        reward = -0.01 # Time penalty

        # --- Action Handling ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        self._handle_input(movement, space_held, shift_held)
        
        # --- Game Logic Updates ---
        self._update_player()
        self._update_enemies()
        self._update_projectiles()
        self._update_particles()

        # --- Collisions and Rewards ---
        collision_rewards = self._handle_collisions()
        reward += collision_rewards

        # --- Heat Reward ---
        current_heat_above_threshold = self.player_heat > self.CAMO_DETECTION_THRESHOLD
        if self.last_heat_above_threshold and not current_heat_above_threshold:
             reward += 0.1 # Reward for cooling down
        self.last_heat_above_threshold = current_heat_above_threshold

        # --- Wave Completion ---
        if not self.enemies:
            reward += 100
            self._spawn_wave()
            # Sound: Wave Complete
        
        # --- Termination Check ---
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS
        if terminated and self.player_health <= 0:
            reward = -100
        
        self.was_space_held = space_held

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, movement, space_held, shift_held):
        # Aiming
        if movement == 1: self.player_angle -= self.PLAYER_AIM_SPEED # Up
        elif movement == 2: self.player_angle += self.PLAYER_AIM_SPEED # Down
        elif movement == 3: self.player_angle -= self.PLAYER_AIM_SPEED # Left
        elif movement == 4: self.player_angle += self.PLAYER_AIM_SPEED # Right
        self.player_angle %= (2 * math.pi)

        # Camouflage
        self.is_camouflaged = shift_held

        # Laser Charging & Firing
        if space_held and not self.is_camouflaged and self.player_fire_cooldown_timer <= 0:
            self.laser_charge = min(100, self.laser_charge + self.LASER_CHARGE_RATE)
        
        if not space_held and self.was_space_held and self.laser_charge >= self.LASER_MIN_CHARGE_TO_FIRE:
            self._fire_laser()
            # Sound: Laser Fire
    
    def _fire_laser(self):
        direction = np.array([math.cos(self.player_angle), math.sin(self.player_angle)])
        start_pos = self.player_pos + direction * 25
        
        bullet = {
            "pos": start_pos,
            "vel": direction * self.LASER_SPEED,
            "charge": self.laser_charge,
            "size": 2 + self.laser_charge / 20,
            "damage": 5 + self.laser_charge * 0.5
        }
        self.player_bullets.append(bullet)
        
        self.player_heat = min(self.PLAYER_MAX_HEAT, self.player_heat + self.laser_charge * 0.4)
        self.laser_charge = 0
        self.player_fire_cooldown_timer = self.PLAYER_FIRE_COOLDOWN

    def _update_player(self):
        if self.player_fire_cooldown_timer > 0:
            self.player_fire_cooldown_timer -= 1
        
        cooldown_rate = self.PLAYER_HEAT_COOLDOWN
        if self.is_camouflaged:
            cooldown_rate += self.PLAYER_CAMO_HEAT_COOLDOWN_BOOST
        
        self.player_heat = max(0, self.player_heat - cooldown_rate)

    def _update_enemies(self):
        for enemy in self.enemies:
            # Movement
            enemy["pos"] += enemy["vel"]
            if enemy["pos"][0] < 50 or enemy["pos"][0] > self.SCREEN_WIDTH - 50:
                enemy["vel"][0] *= -1
            
            # Targeting & Firing
            is_detected = not self.is_camouflaged and self.player_heat > self.CAMO_DETECTION_THRESHOLD
            if is_detected:
                enemy["fire_timer"] -= 1
                if enemy["fire_timer"] <= 0:
                    enemy["fire_timer"] = enemy["fire_rate"]
                    direction = (self.player_pos - enemy["pos"])
                    dist = np.linalg.norm(direction)
                    if dist > 0:
                        direction /= dist
                        start_pos = enemy["pos"] + direction * 20
                        self.enemy_bullets.append({
                            "pos": start_pos,
                            "vel": direction * (self.LASER_SPEED * 0.7)
                        })
                        # Sound: Enemy Laser

    def _update_projectiles(self):
        # Player bullets
        for bullet in self.player_bullets[:]:
            bullet["pos"] += bullet["vel"]
            if not (0 < bullet["pos"][0] < self.SCREEN_WIDTH and 0 < bullet["pos"][1] < self.SCREEN_HEIGHT):
                self.player_bullets.remove(bullet)
        
        # Enemy bullets
        for bullet in self.enemy_bullets[:]:
            bullet["pos"] += bullet["vel"]
            if not (0 < bullet["pos"][0] < self.SCREEN_WIDTH and 0 < bullet["pos"][1] < self.SCREEN_HEIGHT):
                self.enemy_bullets.remove(bullet)

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"] += p["vel"]
            p["life"] -= 1
            if p["life"] <= 0:
                self.particles.remove(p)

    def _handle_collisions(self):
        reward = 0
        
        # Player bullets vs Enemies
        for bullet in self.player_bullets[:]:
            for enemy in self.enemies[:]:
                dist = np.linalg.norm(bullet["pos"] - enemy["pos"])
                if dist < 15 + bullet["size"]: # 15 is enemy radius
                    reward += 1.0 # Hit reward
                    enemy["health"] -= bullet["damage"]
                    self._create_explosion(bullet["pos"], 5, (200, 200, 255), 1)
                    if bullet in self.player_bullets: self.player_bullets.remove(bullet)
                    
                    if enemy["health"] <= 0:
                        reward += 10.0 # Destroy reward
                        self.score += 100
                        self._create_explosion(enemy["pos"], 50, self.COLOR_ENEMY, 3)
                        self.enemies.remove(enemy)
                        # Sound: Explosion
                    break

        # Enemy bullets vs Player
        for bullet in self.enemy_bullets[:]:
            dist = np.linalg.norm(bullet["pos"] - self.player_pos)
            if dist < 15: # 15 is player radius
                self.player_health = max(0, self.player_health - 10)
                self._create_explosion(bullet["pos"], 10, self.COLOR_PLAYER, 1.5)
                self.enemy_bullets.remove(bullet)
                # Sound: Player Hit
        
        return reward

    def _spawn_wave(self):
        self.wave += 1
        num_enemies = 2 + self.wave
        
        for _ in range(num_enemies):
            side = random.choice([0, 1]) # 0 for left/right, 1 for top/bottom
            if side == 0:
                x = random.choice([50, self.SCREEN_WIDTH - 50])
                y = random.uniform(50, self.SCREEN_HEIGHT - 50)
            else:
                x = random.uniform(50, self.SCREEN_WIDTH-50)
                y = random.choice([50, self.SCREEN_HEIGHT - 150])

            enemy_health = self.ENEMY_BASE_HEALTH * (1 + (self.wave - 1) * 0.1)
            enemy_fire_rate = self.ENEMY_BASE_FIRE_RATE * (0.95 ** (self.wave - 1))

            self.enemies.append({
                "pos": np.array([x, y], dtype=np.float32),
                "vel": np.array([random.choice([-1, 1]) * self.ENEMY_SPEED, 0], dtype=np.float32),
                "health": enemy_health,
                "max_health": enemy_health,
                "fire_rate": enemy_fire_rate,
                "fire_timer": random.randint(0, int(enemy_fire_rate))
            })

    def _check_termination(self):
        if self.player_health <= 0:
            self.game_over = True
            self._create_explosion(self.player_pos, 100, self.COLOR_PLAYER, 5)
            # Sound: Player Explosion
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "wave": self.wave}

    def _generate_stars(self, count):
        stars = []
        for _ in range(count):
            x = random.randint(0, self.SCREEN_WIDTH)
            y = random.randint(0, self.SCREEN_HEIGHT)
            size = random.choice([1, 1, 1, 2, 2, 3])
            if size == 1: color, speed = self.COLOR_STAR_FAR, 0.1
            elif size == 2: color, speed = self.COLOR_STAR_MID, 0.3
            else: color, speed = self.COLOR_STAR_NEAR, 0.5
            stars.append({'pos': [x, y], 'size': size, 'color': color, 'speed': speed})
        return stars

    def _create_explosion(self, pos, count, color, speed_scale):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3) * speed_scale
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "life": random.randint(15, 30),
                "color": color,
                "size": random.randint(1, 4)
            })

    # --- Rendering ---
    
    def _render_background(self):
        self.screen.fill(self.COLOR_BG)
        for star in self.stars:
            star['pos'][0] = (star['pos'][0] - star['speed']) % self.SCREEN_WIDTH
            pygame.draw.circle(self.screen, star['color'], star['pos'], star['size'])
    
    def _render_game(self):
        self._render_particles()
        self._render_heat_signature()
        self._render_enemies()
        self._render_player()
        self._render_projectiles()

    def _render_player(self):
        # Shimmer effect for camouflage
        if self.is_camouflaged:
            for i in range(3):
                offset_angle = (self.steps / 10.0 + i * math.pi * 2/3)
                offset_x = math.cos(offset_angle) * 3
                offset_y = math.sin(offset_angle) * 3
                self._draw_ship(self.player_pos + np.array([offset_x, offset_y]), self.player_angle, self.COLOR_CAMO_SHIMMER, alpha=80)

        # Glow
        for i in range(5, 0, -1):
            self._draw_ship(self.player_pos, self.player_angle, self.COLOR_PLAYER_GLOW, scale=1.0 + i * 0.15, alpha_override=True)
        
        # Main ship
        self._draw_ship(self.player_pos, self.player_angle, self.COLOR_PLAYER)
        
        # Aiming reticle
        reticle_dist = 40 + self.laser_charge * 0.4
        reticle_pos = (
            self.player_pos[0] + math.cos(self.player_angle) * reticle_dist,
            self.player_pos[1] + math.sin(self.player_angle) * reticle_dist
        )
        pygame.gfxdraw.aacircle(self.screen, int(reticle_pos[0]), int(reticle_pos[1]), 5, self.COLOR_UI_CHARGE)

    def _draw_ship(self, pos, angle, color, scale=1.0, alpha=255, alpha_override=False):
        points = [
            (15, 0), (-10, -10), (-5, 0), (-10, 10)
        ]
        
        transformed_points = []
        for p in points:
            p_scaled = (p[0] * scale, p[1] * scale)
            x_rot = p_scaled[0] * math.cos(angle) - p_scaled[1] * math.sin(angle)
            y_rot = p_scaled[0] * math.sin(angle) + p_scaled[1] * math.cos(angle)
            transformed_points.append((pos[0] + x_rot, pos[1] + y_rot))
        
        if len(color) == 4 and alpha_override:
            pygame.gfxdraw.aapolygon(self.screen, transformed_points, color)
            pygame.gfxdraw.filled_polygon(self.screen, transformed_points, color)
        else:
            pygame.draw.polygon(self.screen, color, transformed_points)


    def _render_enemies(self):
        for enemy in self.enemies:
            # Glow
            for i in range(3, 0, -1):
                pygame.gfxdraw.filled_circle(self.screen, int(enemy["pos"][0]), int(enemy["pos"][1]), 10 + i * 2, self.COLOR_ENEMY_GLOW)
            
            # Body
            pygame.draw.circle(self.screen, self.COLOR_ENEMY, (int(enemy["pos"][0]), int(enemy["pos"][1])), 10)
            
            # Health bar
            health_pct = max(0, enemy["health"] / enemy["max_health"])
            bar_width = 20
            bar_height = 4
            bar_x = enemy["pos"][0] - bar_width / 2
            bar_y = enemy["pos"][1] - 20
            pygame.draw.rect(self.screen, (50, 0, 0), (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, (bar_x, bar_y, bar_width * health_pct, bar_height))

    def _render_projectiles(self):
        for bullet in self.player_bullets:
            end_pos = bullet["pos"] - bullet["vel"] * 0.5
            pygame.draw.line(self.screen, self.COLOR_PLAYER_LASER, bullet["pos"], end_pos, int(bullet["size"]))
        
        for bullet in self.enemy_bullets:
            pygame.draw.circle(self.screen, self.COLOR_ENEMY_LASER, (int(bullet["pos"][0]), int(bullet["pos"][1])), 3)

    def _render_particles(self):
        for p in self.particles:
            alpha_color = (*p["color"][:3], int(255 * (p["life"] / 30.0)))
            temp_surf = pygame.Surface((p["size"]*2, p["size"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, alpha_color, (p["size"], p["size"]), p["size"])
            self.screen.blit(temp_surf, (int(p["pos"][0] - p["size"]), int(p["pos"][1] - p["size"])), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_heat_signature(self):
        if self.player_heat > 0:
            radius = int(self.player_heat * 0.8)
            alpha = min(255, int(self.player_heat * 1.5))
            color = (*self.COLOR_HEAT_SIG, alpha)
            
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, color)
            self.screen.blit(temp_surf, (self.player_pos[0]-radius, self.player_pos[1]-radius), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Health Bar
        health_pct = self.player_health / self.PLAYER_MAX_HEALTH
        pygame.draw.rect(self.screen, (50, 50, 50), (10, 10, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_UI_HEALTH, (10, 10, 200 * health_pct, 20))
        
        # Heat Gauge
        heat_pct = self.player_heat / self.PLAYER_MAX_HEAT
        pygame.draw.rect(self.screen, (50, 50, 50), (10, 35, 200, 20))
        pygame.draw.rect(self.screen, self.COLOR_UI_HEAT, (10, 35, 200 * heat_pct, 20))
        
        # Laser Charge
        if self.laser_charge > 0:
            charge_pct = self.laser_charge / 100
            pygame.draw.rect(self.screen, (50, 50, 50), (self.player_pos[0] - 50, self.player_pos[1] + 30, 100, 10))
            pygame.draw.rect(self.screen, self.COLOR_UI_CHARGE, (self.player_pos[0] - 50, self.player_pos[1] + 30, 100 * charge_pct, 10))
        
        # Score and Wave Text
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.SCREEN_WIDTH - score_text.get_width() - 10, 10))
        
        wave_text = self.font_large.render(f"WAVE: {self.wave}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.SCREEN_WIDTH/2 - wave_text.get_width()/2, 10))

    def close(self):
        pygame.quit()
    
    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    # It requires pygame to be installed and will open a window.
    # To run headless, this block should be removed or modified.
    os.environ.pop("SDL_VIDEODRIVER", None)
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Starlight Sentinel")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        movement, space, shift = 0, 0, 0
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        if truncated: terminated = True # End loop if truncated
        total_reward += reward
        
        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(30) # Run at 30 FPS
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

    env.close()