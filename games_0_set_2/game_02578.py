
# Generated: 2025-08-27T20:47:43.051245
# Source Brief: brief_02578.md
# Brief Index: 2578

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ←→ to move. Space to fire. Shift to use collected power-up."
    )

    game_description = (
        "Defend Earth from a descending horde of procedurally generated alien invaders in a retro-styled grid-based shooter. Survive 20 waves to win."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_WIDTH, self.GRID_HEIGHT = 12, 10
        self.CELL_SIZE = 36
        self.PLAY_AREA_WIDTH = self.GRID_WIDTH * self.CELL_SIZE
        self.PLAY_AREA_X_OFFSET = (self.WIDTH - self.PLAY_AREA_WIDTH) // 2
        self.PLAYER_Y_POS = self.HEIGHT - 40

        self.MAX_LIVES = 3
        self.MAX_WAVES = 20
        self.MAX_STEPS = 2000
        self.PLAYER_FIRE_COOLDOWN = 4
        self.PLAYER_RAPID_FIRE_COOLDOWN = 2
        self.PLAYER_INVINCIBILITY_FRAMES = 60 # 2 seconds at 30fps

        # --- Colors ---
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_GRID = (30, 25, 50)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_SHIELD = (100, 150, 255, 100)
        self.COLOR_PROJECTILE_PLAYER = (255, 255, 0)
        self.COLOR_PROJECTILE_ALIEN = (255, 80, 80)
        self.COLOR_ALIEN_A = (255, 50, 50)
        self.COLOR_ALIEN_B = (150, 100, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.COLOR_POWERUP_LIFE = (255, 0, 0)
        self.COLOR_POWERUP_SHIELD = (0, 150, 255)
        self.COLOR_POWERUP_RAPID = (255, 255, 0)
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # --- Game State (initialized in reset) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.np_random = None

        self.player_grid_pos = 0
        self.player_lives = 0
        self.player_hit_timer = 0
        self.player_fire_cooldown = 0
        
        self.current_wave = 0
        self.aliens = []
        self.player_projectiles = []
        self.alien_projectiles = []
        self.powerups = []
        self.particles = []

        self.held_powerup = None
        self.active_powerup = None
        self.powerup_timer = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_grid_pos = self.GRID_WIDTH // 2
        self.player_lives = self.MAX_LIVES
        self.player_hit_timer = 0
        self.player_fire_cooldown = 0
        
        self.current_wave = 0
        self.aliens = []
        self.player_projectiles = []
        self.alien_projectiles = []
        self.powerups = []
        self.particles = []

        self.held_powerup = None
        self.active_powerup = None
        self.powerup_timer = 0
        
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0
        
        # --- Handle Input and Update State ---
        reward += self._handle_input(action)
        self._update_game_state()
        reward += self._handle_collisions()
        
        self.steps += 1
        
        # --- Check Game Status ---
        if not self.aliens and not self.win:
            if self.current_wave >= self.MAX_WAVES:
                self.win = True
            else:
                self._spawn_wave()

        terminated = self.player_lives <= 0 or self.win or self.steps >= self.MAX_STEPS
        if terminated and not self.game_over:
            self.game_over = True
            if self.win:
                reward += 100
            elif self.player_lives <= 0:
                reward -= 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, fire_action, powerup_action = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        moved = False
        if movement == 3: # Left
            if self.player_grid_pos > 0:
                self.player_grid_pos -= 1
                moved = True
        elif movement == 4: # Right
            if self.player_grid_pos < self.GRID_WIDTH - 1:
                self.player_grid_pos += 1
                moved = True
        
        if moved:
            reward -= 0.01

        if fire_action and self.player_fire_cooldown == 0:
            # SFX: Player shoot
            px = self.PLAY_AREA_X_OFFSET + self.player_grid_pos * self.CELL_SIZE + self.CELL_SIZE // 2
            self.player_projectiles.append([px, self.PLAYER_Y_POS])
            if self.active_powerup == 'rapid':
                self.player_fire_cooldown = self.PLAYER_RAPID_FIRE_COOLDOWN
            else:
                self.player_fire_cooldown = self.PLAYER_FIRE_COOLDOWN

        if powerup_action and self.held_powerup:
            # SFX: Powerup activate
            self.active_powerup = self.held_powerup
            self.held_powerup = None
            if self.active_powerup == 'shield':
                self.powerup_timer = 300 # 10 seconds
            elif self.active_powerup == 'rapid':
                self.powerup_timer = 300 # 10 seconds
        
        return reward
        
    def _update_game_state(self):
        # Cooldowns and timers
        if self.player_fire_cooldown > 0: self.player_fire_cooldown -= 1
        if self.player_hit_timer > 0: self.player_hit_timer -= 1
        if self.powerup_timer > 0:
            self.powerup_timer -= 1
            if self.powerup_timer == 0:
                self.active_powerup = None
        
        # Projectiles
        self.player_projectiles = [[x, y - 12] for x, y in self.player_projectiles if y > 0]
        self.alien_projectiles = [[x, y + 5] for x, y in self.alien_projectiles if y < self.HEIGHT]

        # Aliens
        alien_speed = 0.25 + (self.current_wave - 1) * 0.05
        alien_fire_prob = 0.005 + (self.current_wave - 1) * 0.002
        for alien in self.aliens:
            alien['pos'][1] += alien_speed
            if self.np_random.random() < alien_fire_prob:
                # SFX: Alien shoot
                self.alien_projectiles.append(list(alien['pixel_pos']))
            
            if alien['pos'][1] * self.CELL_SIZE > self.PLAYER_Y_POS - 20:
                self.aliens.remove(alien)
                self._player_hit()

        # Powerups
        self.powerups = [p for p in self.powerups if p['pos'][1] < self.HEIGHT]
        for p in self.powerups:
            p['pos'][1] += 1

        # Particles
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _handle_collisions(self):
        reward = 0
        player_x = self.PLAY_AREA_X_OFFSET + self.player_grid_pos * self.CELL_SIZE
        player_rect = pygame.Rect(player_x, self.PLAYER_Y_POS, self.CELL_SIZE, 20)

        # Player projectiles vs Aliens
        for proj in self.player_projectiles[:]:
            proj_rect = pygame.Rect(proj[0] - 2, proj[1] - 8, 4, 16)
            for alien in self.aliens[:]:
                alien_rect = pygame.Rect(alien['pixel_pos'][0] - self.CELL_SIZE//2, alien['pixel_pos'][1] - self.CELL_SIZE//2, self.CELL_SIZE, self.CELL_SIZE)
                if proj_rect.colliderect(alien_rect):
                    # SFX: Alien explosion
                    self.player_projectiles.remove(proj)
                    self.aliens.remove(alien)
                    self._create_explosion(alien['pixel_pos'], self.COLOR_ALIEN_A if alien['type'] == 'A' else self.COLOR_ALIEN_B)
                    reward += 1.1 # +1 for kill, +0.1 for hit
                    self.score += 10 * self.current_wave
                    if self.np_random.random() < 0.1: # 10% chance to drop powerup
                        self._spawn_powerup(alien['pixel_pos'])
                    break
        
        # Alien projectiles vs Player
        if self.player_hit_timer == 0:
            for proj in self.alien_projectiles[:]:
                proj_rect = pygame.Rect(proj[0] - 3, proj[1] - 3, 6, 6)
                if proj_rect.colliderect(player_rect):
                    self.alien_projectiles.remove(proj)
                    if self.active_powerup == 'shield':
                        # SFX: Shield block
                        self.active_powerup = None
                        self.powerup_timer = 0
                    else:
                        reward += self._player_hit()
                    break

        # Player vs Powerups
        for p in self.powerups[:]:
            powerup_rect = pygame.Rect(p['pos'][0] - 8, p['pos'][1] - 8, 16, 16)
            if player_rect.colliderect(powerup_rect):
                # SFX: Powerup collect
                self.powerups.remove(p)
                reward += 5
                if p['type'] == 'life':
                    self.player_lives = min(self.MAX_LIVES, self.player_lives + 1)
                else:
                    self.held_powerup = p['type']
                break
        return reward

    def _player_hit(self):
        if self.player_hit_timer > 0: return 0
        # SFX: Player hit/explosion
        self.player_lives -= 1
        self.player_hit_timer = self.PLAYER_INVINCIBILITY_FRAMES
        self._create_explosion([self.PLAY_AREA_X_OFFSET + self.player_grid_pos * self.CELL_SIZE + self.CELL_SIZE // 2, self.PLAYER_Y_POS], self.COLOR_PLAYER)
        return -1

    def _spawn_wave(self):
        self.current_wave += 1
        num_aliens = min(10 + self.current_wave, self.GRID_WIDTH * 3)
        num_rows = (num_aliens + self.GRID_WIDTH -1) // self.GRID_WIDTH
        
        for i in range(num_aliens):
            row = i // self.GRID_WIDTH
            col = i % self.GRID_WIDTH
            
            alien_type = 'B' if self.current_wave > 5 and self.np_random.random() < 0.3 else 'A'
            pos = [col, -row - 1]
            self.aliens.append({'type': alien_type, 'pos': pos})

    def _spawn_powerup(self, pos):
        ptype = self.np_random.choice(['life', 'shield', 'rapid'])
        self.powerups.append({'type': ptype, 'pos': list(pos)})

    def _create_explosion(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            life = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': life, 'color': color})
            
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "wave": self.current_wave,
            "held_powerup": self.held_powerup,
            "active_powerup": self.active_powerup,
        }

    def _render_background(self):
        for x in range(self.GRID_WIDTH + 1):
            px = self.PLAY_AREA_X_OFFSET + x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.HEIGHT))
        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (self.PLAY_AREA_X_OFFSET, py), (self.PLAY_AREA_X_OFFSET + self.PLAY_AREA_WIDTH, py))

    def _render_game(self):
        # Update pixel positions for rendering
        for alien in self.aliens:
            alien['pixel_pos'] = [
                self.PLAY_AREA_X_OFFSET + alien['pos'][0] * self.CELL_SIZE + self.CELL_SIZE / 2,
                alien['pos'][1] * self.CELL_SIZE + self.CELL_SIZE / 2
            ]

        # Draw particles
        for p in self.particles:
            size = max(0, p['life'] / 10)
            pygame.draw.circle(self.screen, p['color'], (int(p['pos'][0]), int(p['pos'][1])), int(size))

        # Draw powerups
        for p in self.powerups:
            color = self.COLOR_POWERUP_LIFE if p['type'] == 'life' else (self.COLOR_POWERUP_SHIELD if p['type'] == 'shield' else self.COLOR_POWERUP_RAPID)
            pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 8, color)
            pygame.gfxdraw.aacircle(self.screen, int(p['pos'][0]), int(p['pos'][1]), 8, color)

        # Draw projectiles
        for x, y in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PROJECTILE_PLAYER, (int(x - 2), int(y - 8), 4, 16))
        for x, y in self.alien_projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(x), int(y), 4, self.COLOR_PROJECTILE_ALIEN)
            pygame.gfxdraw.aacircle(self.screen, int(x), int(y), 4, self.COLOR_PROJECTILE_ALIEN)

        # Draw aliens
        for alien in self.aliens:
            self._draw_alien(alien)

        # Draw player
        if self.player_lives > 0:
            self._draw_player()

    def _draw_player(self):
        is_invincible = self.player_hit_timer > 0
        if is_invincible and (self.steps % 6 < 3):
            return # Flashing effect

        px = self.PLAY_AREA_X_OFFSET + self.player_grid_pos * self.CELL_SIZE
        ship_center_x = px + self.CELL_SIZE // 2
        
        points = [
            (ship_center_x, self.PLAYER_Y_POS - 10),
            (px + 5, self.PLAYER_Y_POS + 10),
            (px + self.CELL_SIZE - 5, self.PLAYER_Y_POS + 10)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)
        pygame.draw.aaline(self.screen, self.COLOR_PLAYER, points[0], points[1])
        pygame.draw.aaline(self.screen, self.COLOR_PLAYER, points[1], points[2])
        pygame.draw.aaline(self.screen, self.COLOR_PLAYER, points[2], points[0])

        if self.active_powerup == 'shield':
            pygame.gfxdraw.aacircle(self.screen, int(ship_center_x), int(self.PLAYER_Y_POS), self.CELL_SIZE // 2 + 2, self.COLOR_PLAYER_SHIELD)
            pygame.gfxdraw.filled_circle(self.screen, int(ship_center_x), int(self.PLAYER_Y_POS), self.CELL_SIZE // 2 + 2, self.COLOR_PLAYER_SHIELD)
    
    def _draw_alien(self, alien):
        x, y = int(alien['pixel_pos'][0]), int(alien['pixel_pos'][1])
        s = self.CELL_SIZE // 2
        color = self.COLOR_ALIEN_A if alien['type'] == 'A' else self.COLOR_ALIEN_B
        
        if alien['type'] == 'A':
            pygame.draw.rect(self.screen, color, (x - s//2, y - s//2, s, s))
            eye_color = (255, 255, 255)
            pygame.draw.rect(self.screen, eye_color, (x - s//4, y - s//4, 2, 2))
            pygame.draw.rect(self.screen, eye_color, (x + s//4 - 2, y - s//4, 2, 2))
        else: # Type B
            points = [ (x, y - s//2), (x - s//2, y), (x, y + s//2), (x + s//2, y) ]
            pygame.draw.polygon(self.screen, color, points)
            pygame.draw.aaline(self.screen, color, points[0], points[1])
            pygame.draw.aaline(self.screen, color, points[1], points[2])
            pygame.draw.aaline(self.screen, color, points[2], points[3])
            pygame.draw.aaline(self.screen, color, points[3], points[0])

        # Simple animation
        if self.steps % 40 > 20:
             pygame.draw.rect(self.screen, (0,0,0), (x-1, y, 2, 2))

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Wave
        wave_text = self.font_small.render(f"WAVE: {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))

        # Lives
        for i in range(self.player_lives):
            points = [
                (20 + i * 25, self.HEIGHT - 20),
                (10 + i * 25, self.HEIGHT - 5),
                (30 + i * 25, self.HEIGHT - 5)
            ]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)

        # Powerup status
        if self.held_powerup:
            p_text = self.font_small.render("STORED:", True, self.COLOR_TEXT)
            self.screen.blit(p_text, (self.WIDTH - 150, self.HEIGHT - 25))
            color = self.COLOR_POWERUP_LIFE if self.held_powerup == 'life' else (self.COLOR_POWERUP_SHIELD if self.held_powerup == 'shield' else self.COLOR_POWERUP_RAPID)
            pygame.gfxdraw.filled_circle(self.screen, self.WIDTH - 70, self.HEIGHT - 18, 8, color)
        
        if self.active_powerup:
            p_text = self.font_small.render(f"ACTIVE: {self.active_powerup.upper()}", True, self.COLOR_TEXT)
            self.screen.blit(p_text, (self.WIDTH / 2 - p_text.get_width()/2, 10))

        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.win else "GAME OVER"
            color = (0, 255, 0) if self.win else (255, 0, 0)
            end_text = self.font_large.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    # To play manually, you'll need to install pygame and run this script.
    # The environment will be rendered to a pygame window.
    
    try:
        import pygame
        from pygame.locals import K_LEFT, K_RIGHT, K_SPACE, K_LSHIFT, K_ESCAPE, K_r
    except ImportError:
        print("Pygame not found. Skipping manual play.")
        exit()

    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Grid-Based Shooter")
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    while not done:
        # Action defaults
        movement = 0 # No-op
        fire = 0 # Released
        powerup = 0 # Released

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == K_ESCAPE):
                done = True

        keys = pygame.key.get_pressed()
        if keys[K_LEFT]:
            movement = 3
        elif keys[K_RIGHT]:
            movement = 4
        
        if keys[K_SPACE]:
            fire = 1
        
        if keys[K_LSHIFT]:
            powerup = 1
            
        if keys[K_r]:
            obs, info = env.reset()

        action = [movement, fire, powerup]
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            # Wait for a moment before auto-resetting
            pygame.time.wait(2000)
            obs, info = env.reset()

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS for consistent game speed

    pygame.quit()