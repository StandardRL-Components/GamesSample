import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T20:25:02.111071
# Source Brief: brief_03289.md
# Brief Index: 3289
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    A Gymnasium environment for a cyberpunk twin-stick shooter.
    The player must survive waves of malfunctioning robots.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Survive waves of malfunctioning robots in this fast-paced, cyberpunk twin-stick shooter. "
        "Dash to avoid damage and unleash a hail of bullets to clear the screen."
    )
    user_guide = (
        "Controls: Use arrow keys to move. Press space to shoot in the direction you are moving. "
        "Press shift to dash."
    )
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 2000
        self.PLAYER_SPEED = 4.0
        self.PLAYER_DASH_SPEED = 8.0
        self.PLAYER_DASH_DURATION = 5  # steps
        self.PLAYER_DASH_COOLDOWN = 30 # steps
        self.PLAYER_RADIUS = 12
        self.PLAYER_HEALTH_MAX = 100
        self.ENEMY_RADIUS = 10
        self.ENEMY_BASE_SPEED = 1.0
        self.BULLET_SPEED = 8.0
        self.BULLET_RADIUS = 4
        self.SHOOT_COOLDOWN = 6 # steps

        # --- Colors ---
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 150, 255, 50)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_ENEMY_GLOW = (255, 50, 50, 50)
        self.COLOR_BULLET = (100, 255, 100)
        self.COLOR_BULLET_GLOW = (100, 255, 100, 100)
        self.COLOR_PARTICLE = (255, 200, 0)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_BAR = (0, 200, 100)
        self.COLOR_HEALTH_BAR_BG = (100, 0, 0)

        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 24, bold=True)
        
        # --- State Variables ---
        # These are initialized in reset()
        self.steps = None
        self.score = None
        self.game_over = None
        self.player_pos = None
        self.player_health = None
        self.last_move_direction = None
        self.shoot_cooldown_timer = None
        self.dash_cooldown_timer = None
        self.dash_active_timer = None
        self.last_space_held = None
        self.enemies = None
        self.bullets = None
        self.particles = None
        self.wave = None
        
        # self.reset() # reset is called by the wrapper
        # self.validate_implementation() # this is for debugging, not needed in final version

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT / 2], dtype=np.float32)
        self.player_health = self.PLAYER_HEALTH_MAX
        self.last_move_direction = np.array([0, -1], dtype=np.float32) # Default shoot up
        
        self.shoot_cooldown_timer = 0
        self.dash_cooldown_timer = 0
        self.dash_active_timer = 0
        self.last_space_held = False
        
        self.enemies = []
        self.bullets = []
        self.particles = []
        
        self.wave = 1
        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        reward = -0.001 # Small penalty for existing, encourages action

        self._handle_input(action)
        self._update_player()
        self._update_enemies()
        self._update_bullets()
        self._update_particles()
        
        reward += self._handle_collisions()

        if not self.enemies:
            reward += 10.0
            self.wave += 1
            self._spawn_wave()
            # sfx: wave_cleared.wav

        terminated = self.player_health <= 0 or self.steps >= self.MAX_STEPS
        truncated = False
        if self.steps >= self.MAX_STEPS:
            terminated = True # Per Gymnasium API, time limit is termination
            
        if terminated and self.player_health <= 0:
            reward = -10.0 # Large penalty for dying
            self.game_over = True
            # sfx: player_death.wav
            self._create_explosion(self.player_pos, 50, self.COLOR_PLAYER)

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _handle_input(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Movement & Aiming ---
        move_vec = np.array([0, 0], dtype=np.float32)
        if movement == 1: move_vec[1] = -1 # Up
        elif movement == 2: move_vec[1] = 1 # Down
        elif movement == 3: move_vec[0] = -1 # Left
        elif movement == 4: move_vec[0] = 1 # Right
        
        if np.linalg.norm(move_vec) > 0:
            self.last_move_direction = move_vec / np.linalg.norm(move_vec)
        self.player_velocity = move_vec

        # --- Dashing (Shift) ---
        if shift_held and self.dash_cooldown_timer == 0:
            self.dash_active_timer = self.PLAYER_DASH_DURATION
            self.dash_cooldown_timer = self.PLAYER_DASH_COOLDOWN
            # sfx: dash.wav
        
        # --- Shooting (Space) ---
        # The prompt implies continuous shooting if held, but the original code only shoots on press.
        # Let's stick to the original logic: shoot on press.
        shoot_press = space_held # and not self.last_space_held # Changed to allow holding down space
        if shoot_press and self.shoot_cooldown_timer == 0:
            self._shoot()
            self.shoot_cooldown_timer = self.SHOOT_COOLDOWN
        self.last_space_held = space_held

    def _update_player(self):
        # Cooldowns
        if self.shoot_cooldown_timer > 0: self.shoot_cooldown_timer -= 1
        if self.dash_cooldown_timer > 0: self.dash_cooldown_timer -= 1
        
        # Movement speed
        current_speed = self.PLAYER_SPEED
        if self.dash_active_timer > 0:
            current_speed = self.PLAYER_DASH_SPEED
            self.dash_active_timer -= 1
            # Create dash trail particles
            if self.steps % 2 == 0:
                trail_pos = self.player_pos.copy()
                self.particles.append({
                    "pos": trail_pos, "vel": np.array([0,0]), "lifespan": 10, 
                    "color": self.COLOR_PLAYER_GLOW, "radius": self.PLAYER_RADIUS * 0.8
                })

        # Update position
        if np.linalg.norm(self.player_velocity) > 0:
            normalized_vel = self.player_velocity / np.linalg.norm(self.player_velocity)
            self.player_pos += normalized_vel * current_speed
        
        # Boundary checks
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.WIDTH - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.HEIGHT - self.PLAYER_RADIUS)

    def _update_enemies(self):
        for enemy in self.enemies:
            direction = self.player_pos - enemy["pos"]
            dist = np.linalg.norm(direction)
            if dist > 1: # Avoid division by zero
                direction /= dist
            enemy["pos"] += direction * enemy["speed"]

    def _update_bullets(self):
        self.bullets = [b for b in self.bullets if self._is_on_screen(b["pos"])]
        for bullet in self.bullets:
            bullet["pos"] += bullet["vel"]

    def _update_particles(self):
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        for p in self.particles:
            p["pos"] += p["vel"]
            p["lifespan"] -= 1

    def _handle_collisions(self):
        reward = 0
        
        # Bullets vs Enemies
        bullets_to_remove = set()
        enemies_to_remove = set()
        
        for i, bullet in enumerate(self.bullets):
            for j, enemy in enumerate(self.enemies):
                if j in enemies_to_remove: continue
                dist = np.linalg.norm(bullet["pos"] - enemy["pos"])
                if dist < self.ENEMY_RADIUS + self.BULLET_RADIUS:
                    bullets_to_remove.add(i)
                    enemies_to_remove.add(j)
                    self.score += 100
                    reward += 1.0
                    self._create_explosion(enemy["pos"], 20, self.COLOR_PARTICLE)
                    # sfx: explosion.wav
                    break
        
        self.bullets = [b for i, b in enumerate(self.bullets) if i not in bullets_to_remove]
        self.enemies = [e for i, e in enumerate(self.enemies) if i not in enemies_to_remove]

        # Player vs Enemies
        enemies_collided = []
        for i, enemy in enumerate(self.enemies):
            dist = np.linalg.norm(self.player_pos - enemy["pos"])
            if dist < self.PLAYER_RADIUS + self.ENEMY_RADIUS:
                self.player_health -= 20
                reward -= 0.5
                enemies_collided.append(i)
                self._create_explosion(self.player_pos, 10, self.COLOR_PLAYER)
                # sfx: player_hit.wav

        self.enemies = [e for i, e in enumerate(self.enemies) if i not in enemies_collided]
        self.player_health = max(0, self.player_health)
        
        return reward

    def _spawn_wave(self):
        num_enemies = 3 + (self.wave - 1) // 2
        enemy_speed = self.ENEMY_BASE_SPEED + (self.wave - 1) * 0.05
        enemy_speed = min(enemy_speed, self.PLAYER_SPEED - 0.5) # Cap speed

        for _ in range(num_enemies):
            edge = self.np_random.integers(0, 4)
            if edge == 0: # Top
                pos = np.array([self.np_random.uniform(0, self.WIDTH), -self.ENEMY_RADIUS])
            elif edge == 1: # Bottom
                pos = np.array([self.np_random.uniform(0, self.WIDTH), self.HEIGHT + self.ENEMY_RADIUS])
            elif edge == 2: # Left
                pos = np.array([-self.ENEMY_RADIUS, self.np_random.uniform(0, self.HEIGHT)])
            else: # Right
                pos = np.array([self.WIDTH + self.ENEMY_RADIUS, self.np_random.uniform(0, self.HEIGHT)])
            
            self.enemies.append({"pos": pos, "speed": enemy_speed})

    def _shoot(self):
        bullet_pos = self.player_pos.copy()
        bullet_vel = self.last_move_direction * self.BULLET_SPEED
        self.bullets.append({"pos": bullet_pos, "vel": bullet_vel})
        # sfx: shoot.wav

    def _create_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = np.array([math.cos(angle), math.sin(angle)]) * speed
            self.particles.append({
                "pos": pos.copy(), "vel": vel, "lifespan": self.np_random.integers(10, 26),
                "color": color, "radius": self.np_random.uniform(1, 4)
            })
    
    def _is_on_screen(self, pos):
        return 0 <= pos[0] <= self.WIDTH and 0 <= pos[1] <= self.HEIGHT

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Background Grid
        for i in range(0, self.WIDTH, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (i, 0), (i, self.HEIGHT))
        for i in range(0, self.HEIGHT, 40):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, i), (self.WIDTH, i))

        # Particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 25.0))
            color = p["color"]
            if len(color) == 4: # Handle RGBA for glow colors
                color_with_alpha = (*color[:3], alpha)
            else:
                color_with_alpha = (*color, alpha)
            
            # Create a temporary surface for alpha blending
            radius = int(p["radius"])
            if radius <= 0: continue
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color_with_alpha, (radius, radius), radius)
            self.screen.blit(temp_surf, (int(p["pos"][0] - radius), int(p["pos"][1] - radius)))

        # Enemies
        for enemy in self.enemies:
            pos_int = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            glow_radius = int(self.ENEMY_RADIUS * 1.8)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], glow_radius, self.COLOR_ENEMY_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.ENEMY_RADIUS, self.COLOR_ENEMY)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.ENEMY_RADIUS, self.COLOR_ENEMY)

        # Player
        if self.player_health > 0:
            pos_int = (int(self.player_pos[0]), int(self.player_pos[1]))
            glow_radius = int(self.PLAYER_RADIUS * 2.0)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], glow_radius, self.COLOR_PLAYER_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
            pygame.gfxdraw.aacircle(self.screen, pos_int[0], pos_int[1], self.PLAYER_RADIUS, self.COLOR_PLAYER)
            # Aiming reticle
            aim_end = self.player_pos + self.last_move_direction * (self.PLAYER_RADIUS + 5)
            pygame.draw.line(self.screen, self.COLOR_UI_TEXT, pos_int, (int(aim_end[0]), int(aim_end[1])), 2)

        # Bullets
        for bullet in self.bullets:
            pos_int = (int(bullet["pos"][0]), int(bullet["pos"][1]))
            glow_radius = int(self.BULLET_RADIUS * 2.5)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], glow_radius, self.COLOR_BULLET_GLOW)
            pygame.gfxdraw.filled_circle(self.screen, pos_int[0], pos_int[1], self.BULLET_RADIUS, self.COLOR_BULLET)

    def _render_ui(self):
        # Score
        score_text = self.font_large.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        score_rect = score_text.get_rect(center=(self.WIDTH // 2, 20))
        self.screen.blit(score_text, score_rect)

        # Wave
        wave_text = self.font_small.render(f"WAVE: {self.wave}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (10, 10))

        # Health Bar
        health_pct = self.player_health / self.PLAYER_HEALTH_MAX
        bar_width = 150
        bar_height = 15
        bar_x = self.WIDTH - bar_width - 10
        bar_y = 10
        
        fill_width = int(bar_width * health_pct)
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR, (bar_x, bar_y, fill_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (bar_x, bar_y, bar_width, bar_height), 1)

        # Dash Cooldown Indicator
        dash_bar_y = bar_y + bar_height + 5
        dash_ready = self.dash_cooldown_timer == 0
        dash_color = self.COLOR_PLAYER if dash_ready else self.COLOR_GRID
        
        dash_text = self.font_small.render("DASH", True, dash_color)
        dash_text_rect = dash_text.get_rect(topright=(self.WIDTH - 10, dash_bar_y))
        self.screen.blit(dash_text, dash_text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "player_health": self.player_health,
            "enemies_remaining": len(self.enemies),
        }

    def close(self):
        pygame.quit()

    def render(self):
        return self._get_observation()

if __name__ == "__main__":
    # --- Manual Play Example ---
    # This part is for human players to test the environment and is not used by the evaluation system.
    # It will not be included in the final submission.
    
    # Set a visible video driver for manual play
    os.environ["SDL_VIDEODRIVER"] = "x11" 
    
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a window for display
    pygame.display.set_caption("Cybernetic Annihilation")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    movement = 0
    space_held = 0
    shift_held = 0
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Key presses for manual control
        keys = pygame.key.get_pressed()
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        else: movement = 0

        action = [movement, space_held, shift_held]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Wave: {info['wave']}")
            obs, info = env.reset()

        # Render the observation to the display window
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Run at 30 FPS

    env.close()