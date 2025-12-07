
# Generated: 2025-08-27T13:50:39.570486
# Source Brief: brief_00502.md
# Brief Index: 502

        
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
        "Controls: ↑↓←→ to move. Hold Space to fire. Good luck, pilot!"
    )

    game_description = (
        "Pilot a spaceship in a top-down shooter to destroy waves of procedurally generated alien invaders."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.PLAYER_SPEED = 6
        self.PLAYER_HEALTH_MAX = 100
        self.PLAYER_FIRE_COOLDOWN = 5  # 6 shots/sec @ 30fps
        self.PROJECTILE_SPEED = 10
        self.TOTAL_ALIENS_PER_WAVE = 50
        self.MAX_STEPS = 3000

        # --- Colors ---
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_GLOW = (100, 255, 200, 100)
        self.COLOR_PLAYER_PROJ = (255, 255, 0)
        self.COLOR_ENEMY_PROJ = (255, 255, 255)
        self.COLOR_ENEMY_RED = (255, 50, 50)
        self.COLOR_ENEMY_BLUE = (100, 150, 255)
        self.COLOR_ENEMY_PURPLE = (200, 100, 255)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_HEALTH_HIGH = (0, 255, 0)
        self.COLOR_HEALTH_MED = (255, 255, 0)
        self.COLOR_HEALTH_LOW = (255, 0, 0)

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
        self.font_ui = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 72)
        
        # Initialize state variables to prevent attribute errors before first reset
        self.steps = 0
        self.score = 0
        self.wave = 1
        self.game_over = False
        self.last_win = False
        self.player_pos = [0,0]
        self.player_health = 0
        self.player_fire_timer = 0
        self.aliens = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []
        self.stars = []

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.last_win:
            self.wave += 1
        else:
            self.wave = 1
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_win = False

        self.player_pos = [self.WIDTH // 2, self.HEIGHT - 50]
        self.player_health = self.PLAYER_HEALTH_MAX
        self.player_fire_timer = 0
        
        self.aliens = []
        self.player_projectiles = []
        self.enemy_projectiles = []
        self.particles = []

        self._spawn_stars()
        self._spawn_aliens()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack action and handle input
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self._handle_player_input(movement, space_held)

        # Update game logic
        self.steps += 1
        reward = -0.01  # Small penalty for each step to encourage efficiency

        aliens_destroyed_this_step = self._update_projectiles()
        reward += aliens_destroyed_this_step * 1.0
        self.score += aliens_destroyed_this_step * 10
        
        self._update_aliens()
        self._update_particles()
        
        # Check termination conditions
        terminated = False
        if self.player_health <= 0:
            terminated = True
            self.game_over = True
            reward = -100.0
            self._create_explosion(self.player_pos, self.COLOR_PLAYER, 100, 2.0)
        elif not self.aliens:
            terminated = True
            self.game_over = True
            self.last_win = True
            reward = 50.0
            self.score += 500 # Wave clear bonus
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True
            reward = -100.0

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _handle_player_input(self, movement, space_held):
        # Movement
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED
        if movement == 2: self.player_pos[1] += self.PLAYER_SPEED
        if movement == 3: self.player_pos[0] -= self.PLAYER_SPEED
        if movement == 4: self.player_pos[0] += self.PLAYER_SPEED

        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.HEIGHT)

        # Firing
        if self.player_fire_timer > 0:
            self.player_fire_timer -= 1
        
        if space_held and self.player_fire_timer == 0:
            # sfx: player_shoot.wav
            proj_pos = [self.player_pos[0], self.player_pos[1] - 20]
            self.player_projectiles.append(proj_pos)
            self.player_fire_timer = self.PLAYER_FIRE_COOLDOWN

    def _update_projectiles(self):
        aliens_destroyed = 0
        
        # Player projectiles
        for proj in self.player_projectiles[:]:
            proj[1] -= self.PROJECTILE_SPEED
            if proj[1] < 0:
                self.player_projectiles.remove(proj)
                continue
            
            for alien in self.aliens[:]:
                dist = math.hypot(proj[0] - alien['pos'][0], proj[1] - alien['pos'][1])
                if dist < 15: # Collision
                    # sfx: explosion.wav
                    self._create_explosion(alien['pos'], alien['color'], 30, 1.0)
                    self.aliens.remove(alien)
                    self.player_projectiles.remove(proj)
                    aliens_destroyed += 1
                    break
        
        # Enemy projectiles
        for proj in self.enemy_projectiles[:]:
            proj[1] += proj[2] # proj[2] is speed
            if proj[1] > self.HEIGHT:
                self.enemy_projectiles.remove(proj)
                continue
            
            dist = math.hypot(proj[0] - self.player_pos[0], proj[1] - self.player_pos[1])
            if dist < 15: # Player hit
                # sfx: player_hit.wav
                self.player_health -= 10
                self._create_explosion(self.player_pos, self.COLOR_PLAYER_GLOW, 10, 0.5)
                self.enemy_projectiles.remove(proj)
                
        return aliens_destroyed

    def _update_aliens(self):
        difficulty_mod = 1 + (self.wave - 1) * 0.05
        base_fire_rate = 0.002
        
        for alien in self.aliens:
            # Movement
            if alien['type'] == 'red':
                alien['pos'][0] = alien['initial_pos'][0] + math.sin(self.steps * 0.03 + alien['phase']) * 60
                alien['pos'][1] += 0.5 * difficulty_mod
            elif alien['type'] == 'blue':
                alien['pos'][0] += alien['vel'][0] * difficulty_mod
                alien['pos'][1] += alien['vel'][1] * difficulty_mod
                if not 0 < alien['pos'][0] < self.WIDTH:
                    alien['vel'][0] *= -1
            elif alien['type'] == 'purple':
                alien['pos'][0] = self.WIDTH/2 + math.sin(self.steps * 0.01 + alien['phase']) * (self.WIDTH/2 - 40)

            # Firing
            fire_chance = base_fire_rate * difficulty_mod
            if self.np_random.random() < fire_chance:
                # sfx: enemy_shoot.wav
                proj_pos = [alien['pos'][0], alien['pos'][1] + 15]
                proj_speed = 4 * difficulty_mod
                self.enemy_projectiles.append([proj_pos[0], proj_pos[1], proj_speed])

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.05 # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)

    def _spawn_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'pos': [self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)],
                'brightness': self.np_random.integers(50, 150)
            })

    def _spawn_aliens(self):
        rows, cols = 5, 10
        for i in range(self.TOTAL_ALIENS_PER_WAVE):
            row = i // cols
            col = i % cols
            x = col * 50 + self.WIDTH/2 - (cols*50)/2 + 25
            y = row * 40 + 50
            
            alien_type = 'red'
            color = self.COLOR_ENEMY_RED
            if row >= 2:
                alien_type = 'blue'
                color = self.COLOR_ENEMY_BLUE
            if row >= 4:
                alien_type = 'purple'
                color = self.COLOR_ENEMY_PURPLE

            self.aliens.append({
                'pos': [x, y],
                'initial_pos': [x, y],
                'type': alien_type,
                'color': color,
                'phase': self.np_random.random() * 2 * math.pi,
                'vel': [(1 if self.np_random.random() > 0.5 else -1) * 1.5, 0.2]
            })

    def _create_explosion(self, pos, color, count, speed_multiplier):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 3 * speed_multiplier
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(15, 30),
                'color': color
            })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_stars()
        self._render_particles()
        self._render_projectiles()
        if not (self.game_over and self.player_health <= 0):
            self._render_player()
        self._render_aliens()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_stars(self):
        for star in self.stars:
            b = star['brightness']
            if self.np_random.random() < 0.01: # Twinkle
                b = self.np_random.integers(50, 150)
                star['brightness'] = b
            pygame.draw.circle(self.screen, (b, b, b), star['pos'], 1)

    def _render_player(self):
        p = self.player_pos
        points = [(p[0], p[1] - 15), (p[0] - 10, p[1] + 10), (p[0] + 10, p[1] + 10)]
        
        # Engine glow
        glow_size = 15 + math.sin(self.steps * 0.5) * 3
        pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]+5), int(glow_size), self.COLOR_PLAYER_GLOW)
        
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
        
    def _render_aliens(self):
        for alien in self.aliens:
            x, y = int(alien['pos'][0]), int(alien['pos'][1])
            color = alien['color']
            if alien['type'] == 'red': # V-shape
                points = [(x, y+10), (x-10, y-5), (x, y), (x+10, y-5)]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
            elif alien['type'] == 'blue': # Diamond
                points = [(x, y-10), (x-8, y), (x, y+10), (x+8, y)]
                pygame.gfxdraw.aapolygon(self.screen, points, color)
                pygame.gfxdraw.filled_polygon(self.screen, points, color)
            elif alien['type'] == 'purple': # Saucer
                pygame.gfxdraw.aaellipse(self.screen, x, y, 12, 6, color)
                pygame.gfxdraw.filled_ellipse(self.screen, x, y, 12, 6, color)
                pygame.gfxdraw.filled_circle(self.screen, x, y-2, 4, (255,255,255,150))


    def _render_projectiles(self):
        for p in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ, (p[0]-2, p[1]-5, 4, 10))
        for p in self.enemy_projectiles:
            pygame.gfxdraw.filled_circle(self.screen, int(p[0]), int(p[1]), 3, self.COLOR_ENEMY_PROJ)
            pygame.gfxdraw.aacircle(self.screen, int(p[0]), int(p[1]), 3, self.COLOR_ENEMY_PROJ)

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(p['life'] * (255/30))))
            color = p['color'][:3]
            size = max(1, int(p['life'] / 10))
            pygame.draw.circle(self.screen, color, p['pos'], size)

    def _render_ui(self):
        # Health Bar
        health_ratio = self.player_health / self.PLAYER_HEALTH_MAX
        health_color = self.COLOR_HEALTH_LOW
        if health_ratio > 0.66: health_color = self.COLOR_HEALTH_HIGH
        elif health_ratio > 0.33: health_color = self.COLOR_HEALTH_MED
        
        pygame.draw.rect(self.screen, (50,50,50), (10, 10, 200, 20))
        pygame.draw.rect(self.screen, health_color, (10, 10, 200 * health_ratio, 20))
        
        # Score and Wave
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 10))
        
        wave_text = self.font_ui.render(f"WAVE: {self.wave}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.WIDTH/2 - wave_text.get_width()/2, 10))
        
        # Game Over Message
        if self.game_over:
            msg = "WAVE CLEARED" if self.last_win else "GAME OVER"
            color = self.COLOR_PLAYER if self.last_win else self.COLOR_ENEMY_RED
            over_text = self.font_game_over.render(msg, True, color)
            text_rect = over_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            pygame.draw.rect(self.screen, (0,0,0,150), text_rect.inflate(20, 20))
            self.screen.blit(over_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "player_health": self.player_health,
            "aliens_remaining": len(self.aliens)
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
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Arcade Space Shooter")
    clock = pygame.time.Clock()
    running = True
    
    total_reward = 0
    
    while running:
        # --- Action mapping for human play ---
        keys = pygame.key.get_pressed()
        movement = 0 # none
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Gym step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already a rendered frame, so we just need to display it
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if terminated or truncated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
            # Wait for a moment before resetting
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(30) # Run at 30 FPS

    env.close()