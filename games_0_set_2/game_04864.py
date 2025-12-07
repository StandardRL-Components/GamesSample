
# Generated: 2025-08-28T03:14:57.075617
# Source Brief: brief_04864.md
# Brief Index: 4864

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Hold Space to shoot. Hold Shift for a temporary shield."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down space shooter. Destroy waves of aliens while dodging their projectiles."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.width, self.height = 640, 400
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # Colors
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_SHIELD = (100, 150, 255, 100)
        self.COLOR_PLAYER_PROJ = (200, 255, 255)
        self.COLOR_ENEMY_PROJ = (255, 200, 0)
        self.WAVE_COLORS = {
            1: (255, 80, 80),  # Red
            2: (80, 120, 255), # Blue
            3: (200, 80, 255), # Purple
            4: (255, 150, 50), # Orange
            5: (240, 240, 240) # White
        }
        self.COLOR_EXPLOSION = (255, 128, 0)
        self.COLOR_UI_TEXT = (220, 220, 220)

        # Fonts
        try:
            self.font_s = pygame.font.SysFont("Consolas", 18)
            self.font_l = pygame.font.SysFont("Consolas", 48)
        except pygame.error:
            self.font_s = pygame.font.Font(None, 24)
            self.font_l = pygame.font.Font(None, 60)

        # Game Constants
        self.PLAYER_SPEED = 6
        self.PLAYER_SHOOT_COOLDOWN_MAX = 6 # frames
        self.PLAYER_SHIELD_DURATION = 15 # frames
        self.PLAYER_RADIUS = 12
        self.PROJECTILE_RADIUS = 4
        self.ENEMY_RADIUS = 10
        self.MAX_WAVES = 5
        self.MAX_STEPS = 3000

        # Initialize state variables
        self.player_pos = None
        self.player_health = None
        self.player_projectiles = None
        self.player_shoot_cooldown = None
        self.player_shield_timer = None
        self.enemies = None
        self.enemy_projectiles = None
        self.particles = None
        self.stars = None
        self.score = None
        self.steps = None
        self.current_wave = None
        self.wave_cleared = None
        self.game_over = None
        self.win = None
        
        # Initialize state variables
        self.reset()
        
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize all game state
        self.player_pos = [self.width // 2, self.height - 50]
        self.player_health = 100

        self.player_projectiles = []
        self.player_shoot_cooldown = 0
        self.player_shield_timer = 0

        self.enemies = []
        self.enemy_projectiles = []
        self.particles = []

        self.score = 0
        self.steps = 0
        self.current_wave = 1
        self.wave_cleared = True # Triggers first wave spawn
        self.game_over = False
        self.win = False

        if self.stars is None: # Only create stars once
            self.stars = [
                {
                    "pos": [self.np_random.integers(0, self.width), self.np_random.integers(0, self.height)],
                    "speed": self.np_random.uniform(0.5, 1.5),
                    "size": self.np_random.integers(1, 3)
                } for _ in range(100)
            ]
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        # --- Handle Wave Spawning ---
        if self.wave_cleared and not self.enemies:
            if self.current_wave > self.MAX_WAVES:
                self.win = True
            else:
                self._spawn_wave()
                self.wave_cleared = False

        # --- Handle Input & Player Update ---
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1

        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED # Up
        if movement == 2: self.player_pos[1] += self.PLAYER_SPEED # Down
        if movement == 3: self.player_pos[0] -= self.PLAYER_SPEED # Left
        if movement == 4: self.player_pos[0] += self.PLAYER_SPEED # Right

        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_RADIUS, self.width - self.PLAYER_RADIUS)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_RADIUS, self.height - self.PLAYER_RADIUS)

        if self.player_shoot_cooldown > 0: self.player_shoot_cooldown -= 1
        if space_held and self.player_shoot_cooldown == 0:
            # SFX: Player shoot
            self.player_projectiles.append({
                "pos": [self.player_pos[0], self.player_pos[1] - self.PLAYER_RADIUS],
                "speed": -12
            })
            self.player_shoot_cooldown = self.PLAYER_SHOOT_COOLDOWN_MAX

        if self.player_shield_timer > 0: self.player_shield_timer -= 1
        if shift_held and self.player_shield_timer == 0:
            # SFX: Shield activate
            self.player_shield_timer = self.PLAYER_SHIELD_DURATION

        # --- Update Game Objects ---
        self._update_projectiles()
        self._update_enemies()
        self._update_particles()
        
        # --- Handle Collisions ---
        reward += self._handle_collisions()

        # --- Check Game State ---
        if not self.enemies and not self.win:
            self.wave_cleared = True
            if self.current_wave <= self.MAX_WAVES:
                 self.current_wave += 1

        terminated = False
        if self.player_health <= 0:
            reward -= 10
            self.game_over = True
            terminated = True
            self._create_explosion(self.player_pos, 30, self.COLOR_PLAYER)
            # SFX: Player explosion
        
        if self.win:
            reward += 100
            self.game_over = True
            terminated = True

        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
        
        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _spawn_wave(self):
        num_enemies_map = {1: 5, 2: 7, 3: 9, 4: 3, 5: 10}
        health_map = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        
        num_enemies = num_enemies_map[self.current_wave]
        
        for i in range(num_enemies):
            enemy = {
                "health": health_map[self.current_wave],
                "shoot_cooldown": self.np_random.integers(60, 120),
                "pattern": self.current_wave,
            }
            if enemy["pattern"] == 1: # Horizontal
                enemy["pos"] = [i * (self.width / num_enemies) + 50, 60]
                enemy["speed"] = self.np_random.choice([-2, 2])
            elif enemy["pattern"] == 2: # Vertical
                enemy["pos"] = [i * (self.width / num_enemies) + 50, 60]
                enemy["speed"] = 2
            elif enemy["pattern"] == 3: # Diagonal
                enemy["pos"] = [self.np_random.integers(50, self.width - 50), 60]
                enemy["vel"] = [self.np_random.choice([-3, 3]), 3]
            elif enemy["pattern"] == 4: # Circular
                enemy["center"] = [self.width/2, 120]
                enemy["angle"] = (i / num_enemies) * 2 * math.pi
                enemy["radius"] = 100
                enemy["angular_speed"] = 0.03
                enemy["pos"] = [0,0] # will be set in update
            elif enemy["pattern"] == 5: # Random Walk
                enemy["pos"] = [self.np_random.integers(50, self.width-50), self.np_random.integers(50, 150)]
                enemy["target"] = [self.np_random.integers(50, self.width-50), self.np_random.integers(50, 150)]

            self.enemies.append(enemy)

    def _update_enemies(self):
        for enemy in self.enemies:
            # Movement
            if enemy["pattern"] == 1: # Horizontal
                enemy["pos"][0] += enemy["speed"]
                if not (self.ENEMY_RADIUS < enemy["pos"][0] < self.width - self.ENEMY_RADIUS):
                    enemy["speed"] *= -1
            elif enemy["pattern"] == 2: # Vertical
                enemy["pos"][1] += enemy["speed"]
                if not (40 < enemy["pos"][1] < self.height / 2):
                    enemy["speed"] *= -1
            elif enemy["pattern"] == 3: # Diagonal
                enemy["pos"][0] += enemy["vel"][0]
                enemy["pos"][1] += enemy["vel"][1]
                if not (self.ENEMY_RADIUS < enemy["pos"][0] < self.width - self.ENEMY_RADIUS):
                    enemy["vel"][0] *= -1
                if not (self.ENEMY_RADIUS < enemy["pos"][1] < self.height - self.ENEMY_RADIUS):
                    enemy["vel"][1] *= -1
            elif enemy["pattern"] == 4: # Circular
                enemy["angle"] += enemy["angular_speed"]
                enemy["pos"][0] = enemy["center"][0] + math.cos(enemy["angle"]) * enemy["radius"]
                enemy["pos"][1] = enemy["center"][1] + math.sin(enemy["angle"]) * enemy["radius"]
            elif enemy["pattern"] == 5: # Random Walk
                dist = math.hypot(enemy["target"][0] - enemy["pos"][0], enemy["target"][1] - enemy["pos"][1])
                if dist < 10:
                    enemy["target"] = [self.np_random.integers(50, self.width-50), self.np_random.integers(50, 150)]
                else:
                    angle = math.atan2(enemy["target"][1] - enemy["pos"][1], enemy["target"][0] - enemy["pos"][0])
                    enemy["pos"][0] += math.cos(angle) * 2
                    enemy["pos"][1] += math.sin(angle) * 2

            # Shooting
            enemy["shoot_cooldown"] -= 1
            if enemy["shoot_cooldown"] <= 0:
                # SFX: Enemy shoot
                proj_speed = 3 + 0.2 * (self.current_wave - 1)
                angle = math.atan2(self.player_pos[1] - enemy["pos"][1], self.player_pos[0] - enemy["pos"][0])
                self.enemy_projectiles.append({
                    "pos": list(enemy["pos"]),
                    "vel": [math.cos(angle) * proj_speed, math.sin(angle) * proj_speed]
                })
                enemy["shoot_cooldown"] = self.np_random.integers(90, 180) - self.current_wave * 10

    def _update_projectiles(self):
        for proj in self.player_projectiles:
            proj["pos"][1] += proj["speed"]
        self.player_projectiles = [p for p in self.player_projectiles if p["pos"][1] > 0]

        for proj in self.enemy_projectiles:
            proj["pos"][0] += proj["vel"][0]
            proj["pos"][1] += proj["vel"][1]
        self.enemy_projectiles = [p for p in self.enemy_projectiles if 0 < p["pos"][0] < self.width and 0 < p["pos"][1] < self.height]

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["lifespan"] -= 1
            p["vel"][1] += 0.05 # a little gravity
        self.particles = [p for p in self.particles if p["lifespan"] > 0]
        
    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs Enemies
        for proj in self.player_projectiles[:]:
            for enemy in self.enemies[:]:
                dist = math.hypot(proj["pos"][0] - enemy["pos"][0], proj["pos"][1] - enemy["pos"][1])
                if dist < self.ENEMY_RADIUS + self.PROJECTILE_RADIUS:
                    reward += 0.1
                    enemy["health"] -= 1
                    if proj in self.player_projectiles: self.player_projectiles.remove(proj)
                    
                    if enemy["health"] <= 0:
                        reward += 1
                        self.score += 1
                        self._create_explosion(enemy["pos"], 15, self.WAVE_COLORS[enemy["pattern"]])
                        self.enemies.remove(enemy)
                        # SFX: Enemy explosion
                    break
        
        # Enemy projectiles vs Player
        is_shielded = self.player_shield_timer > 0
        for proj in self.enemy_projectiles[:]:
            dist = math.hypot(proj["pos"][0] - self.player_pos[0], proj["pos"][1] - self.player_pos[1])
            if dist < self.PLAYER_RADIUS + self.PROJECTILE_RADIUS:
                if is_shielded:
                    # SFX: Shield block
                    self._create_explosion(proj["pos"], 5, self.COLOR_SHIELD[:3])
                else:
                    # SFX: Player hit
                    reward -= 0.1
                    self.player_health -= 10
                    self._create_explosion(self.player_pos, 5, self.COLOR_PLAYER)
                self.enemy_projectiles.remove(proj)

        return reward

    def _create_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * speed, math.sin(angle) * speed],
                "lifespan": self.np_random.integers(15, 30),
                "color": color,
                "radius": self.np_random.uniform(1, 4)
            })

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        # Render all game elements
        self._render_game()
        # Render UI overlay
        self._render_ui()
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stars
        for star in self.stars:
            star["pos"][1] += star["speed"]
            if star["pos"][1] > self.height:
                star["pos"] = [self.np_random.integers(0, self.width), 0]
            pygame.draw.circle(self.screen, (200, 200, 220), star["pos"], star["size"])
            
        # Particles
        for p in self.particles:
            alpha = int(255 * (p["lifespan"] / 30))
            color = (*p["color"], alpha)
            temp_surf = pygame.Surface((p["radius"]*2, p["radius"]*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p["radius"], p["radius"]), p["radius"])
            self.screen.blit(temp_surf, (int(p["pos"][0] - p["radius"]), int(p["pos"][1] - p["radius"])))

        # Enemy Projectiles
        for proj in self.enemy_projectiles:
            pos = (int(proj["pos"][0]), int(proj["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PROJECTILE_RADIUS, self.COLOR_ENEMY_PROJ)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PROJECTILE_RADIUS, self.COLOR_ENEMY_PROJ)

        # Player Projectiles
        for proj in self.player_projectiles:
            pos = (int(proj["pos"][0]), int(proj["pos"][1]))
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PROJECTILE_RADIUS, self.COLOR_PLAYER_PROJ)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PROJECTILE_RADIUS, self.COLOR_PLAYER_PROJ)

        # Enemies
        for enemy in self.enemies:
            pos = (int(enemy["pos"][0]), int(enemy["pos"][1]))
            color = self.WAVE_COLORS[enemy["pattern"]]
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.ENEMY_RADIUS, color)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.ENEMY_RADIUS, color)

        # Player
        if self.player_health > 0:
            pos = (int(self.player_pos[0]), int(self.player_pos[1]))
            
            # Shield effect
            if self.player_shield_timer > 0:
                shield_radius = self.PLAYER_RADIUS + 5 + math.sin(self.steps * 0.5) * 2
                alpha = 50 + (self.player_shield_timer / self.PLAYER_SHIELD_DURATION) * 100
                color = (*self.COLOR_SHIELD[:3], int(alpha))
                temp_surf = pygame.Surface((shield_radius*2, shield_radius*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (shield_radius, shield_radius), shield_radius)
                self.screen.blit(temp_surf, (pos[0] - shield_radius, pos[1] - shield_radius))

            # Ship body
            p1 = (pos[0], pos[1] - self.PLAYER_RADIUS)
            p2 = (pos[0] - self.PLAYER_RADIUS * 0.8, pos[1] + self.PLAYER_RADIUS * 0.8)
            p3 = (pos[0] + self.PLAYER_RADIUS * 0.8, pos[1] + self.PLAYER_RADIUS * 0.8)
            pygame.gfxdraw.aapolygon(self.screen, (p1, p2, p3), self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, (p1, p2, p3), self.COLOR_PLAYER)

    def _render_ui(self):
        # Health Bar
        health_pct = max(0, self.player_health / 100)
        bar_width = 150
        pygame.draw.rect(self.screen, (80, 0, 0), (10, 10, bar_width, 20))
        pygame.draw.rect(self.screen, (0, 180, 0), (10, 10, int(bar_width * health_pct), 20))
        pygame.draw.rect(self.screen, self.COLOR_UI_TEXT, (10, 10, bar_width, 20), 1)

        # Score
        score_text = self.font_s.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.width - score_text.get_width() - 10, 10))

        # Wave
        wave_text = self.font_s.render(f"WAVE: {min(self.current_wave, self.MAX_WAVES)} / {self.MAX_WAVES}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.width / 2 - wave_text.get_width() / 2, 10))
        
        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            end_text = self.font_l.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.width / 2, self.height / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "player_health": self.player_health,
            "current_wave": self.current_wave,
        }

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.height, self.width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.height, self.width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    import os
    # Set the video driver. 'dummy' for headless, or a display driver to see the game.
    # Common drivers: 'x11', 'dga', 'fbcon', 'directfb', 'ggi', 'vgl', 'svgalib', 'aalib'
    # For Windows, it's usually 'windib', 'directx'.
    # If you are not sure, you can comment this line out.
    if os.name == 'posix':
        os.environ.setdefault('SDL_VIDEODRIVER', 'x11')
    
    env = GameEnv()
    obs, info = env.reset()
    
    try:
        screen = pygame.display.set_mode((env.width, env.height))
        pygame.display.set_caption("Space Shooter")
        clock = pygame.time.Clock()
        
        done = False
        total_reward = 0
        
        while not done:
            # --- Human Controls ---
            movement, space, shift = 0, 0, 0
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
                
            action = [movement, space, shift]
            
            # --- Environment Step ---
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            # --- Rendering ---
            # The observation is the rendered frame, so we just need to show it
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            # --- Event Handling & Frame Rate ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            
            clock.tick(30) # Match the intended FPS

        print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")

    except pygame.error as e:
        print("\nPygame display could not be initialized.")
        print("This is normal if you are running in a headless environment.")
        print("The environment is still functional for training RL agents.")
        print(f"Pygame error: {e}")
    finally:
        env.close()