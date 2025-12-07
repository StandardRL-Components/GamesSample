
# Generated: 2025-08-28T07:04:34.905835
# Source Brief: brief_03127.md
# Brief Index: 3127

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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
        "Controls: Arrow keys to move. Hold space to fire. Avoid enemy ships and projectiles."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fend off waves of descending aliens in this retro-inspired top-down shooter. Clear all 5 waves to win."
    )

    # Frames auto-advance at 30fps for smooth, real-time gameplay.
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STEPS = 30 * 60 # 60 seconds

        # Colors
        self.COLOR_BG = (16, 16, 32) # Dark blue
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_ALIEN = (255, 50, 50)
        self.COLOR_PLAYER_PROJ = (255, 255, 0)
        self.COLOR_ALIEN_PROJ = (255, 0, 255)
        self.COLOR_EXPLOSION = [(255, 165, 0), (255, 255, 0), (255, 255, 255)]
        self.COLOR_STAR = (200, 200, 220)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_HEALTH_FG = (0, 255, 0)
        self.COLOR_HEALTH_BG = (128, 0, 0)
        self.COLOR_HEALTH_BORDER = (64, 64, 64)

        # Player settings
        self.PLAYER_SPEED = 8.0
        self.PLAYER_SIZE = 15
        self.PLAYER_FIRE_COOLDOWN = 4 # steps
        self.PLAYER_MAX_HEALTH = 3

        # Alien settings
        self.ALIEN_SIZE = 12
        self.WAVE_COUNT = 5
        self.ALIEN_H_SPEED_BASE = 1.0
        self.ALIEN_V_SPEED = 0.2
        self.ALIEN_FIRE_RATE_BASE = 0.005

        # Projectile settings
        self.PROJ_SPEED = 12.0
        self.PROJ_SIZE = 3

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 72)

        # --- State Variables ---
        # These are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_game = False
        self.np_random = None
        self.player_pos = None
        self.player_health = None
        self.player_fire_timer = None
        self.aliens = None
        self.player_projectiles = None
        self.alien_projectiles = None
        self.explosions = None
        self.stars = None
        self.current_wave = None
        self.alien_move_direction = None
        self.alien_move_timer = None
        self.reward_this_step = 0

        self.validate_implementation()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_game = False
        self.reward_this_step = 0

        self.player_pos = np.array([self.WIDTH / 2, self.HEIGHT - 50], dtype=np.float32)
        self.player_health = self.PLAYER_MAX_HEALTH
        self.player_fire_timer = 0

        self.aliens = []
        self.player_projectiles = []
        self.alien_projectiles = []
        self.explosions = []
        
        self.stars = [
            (self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT), self.np_random.uniform(0.5, 1.5))
            for _ in range(150)
        ]

        self.current_wave = 0
        self.alien_move_direction = 1
        self.alien_move_timer = 0
        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)
        
        self.reward_this_step = -0.01 # Small penalty for existing

        if not self.game_over:
            self._handle_input(action)
            self._update_player()
            self._update_aliens()
            self._update_projectiles()
            self._handle_collisions()
            self._update_waves()

        self._update_effects()
        self.steps += 1
        
        terminated = self._check_termination()
        
        return (
            self._get_observation(),
            self.reward_this_step,
            terminated,
            False,
            self._get_info()
        )
        
    def _handle_input(self, action):
        movement, space_held, _ = action
        
        # Movement
        if movement == 1: # Up
            self.player_pos[1] -= self.PLAYER_SPEED
        elif movement == 2: # Down
            self.player_pos[1] += self.PLAYER_SPEED
        elif movement == 3: # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos[0] += self.PLAYER_SPEED
        
        # Firing
        if space_held and self.player_fire_timer <= 0:
            # sfx: player_shoot.wav
            self.player_projectiles.append(self.player_pos.copy())
            self.player_fire_timer = self.PLAYER_FIRE_COOLDOWN

    def _update_player(self):
        # Cooldown
        if self.player_fire_timer > 0:
            self.player_fire_timer -= 1
        
        # Boundaries
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE, self.HEIGHT - self.PLAYER_SIZE)

    def _update_aliens(self):
        if not self.aliens:
            return

        move_sideways = False
        if self.alien_move_timer <= 0:
            move_sideways = True
            self.alien_move_timer = 20 # steps
        else:
            self.alien_move_timer -= 1

        speed_multiplier = 1 + self.current_wave * 0.1
        h_speed = self.ALIEN_H_SPEED_BASE * speed_multiplier
        
        switch_direction = False
        for alien in self.aliens:
            if move_sideways:
                alien['pos'][0] += h_speed * self.alien_move_direction
            alien['pos'][1] += self.ALIEN_V_SPEED * speed_multiplier
            
            if alien['pos'][0] < self.ALIEN_SIZE or alien['pos'][0] > self.WIDTH - self.ALIEN_SIZE:
                switch_direction = True
            
            if alien['pos'][1] > self.HEIGHT:
                self.game_over = True # Alien reached bottom
                self.player_health = 0
        
        if switch_direction:
            self.alien_move_direction *= -1

        # Alien Firing
        fire_rate = self.ALIEN_FIRE_RATE_BASE * (1 + self.current_wave * 0.5)
        for alien in self.aliens:
            if self.np_random.random() < fire_rate:
                # sfx: alien_shoot.wav
                self.alien_projectiles.append(alien['pos'].copy())

    def _update_projectiles(self):
        self.player_projectiles = [p for p in self.player_projectiles if p[1] > 0]
        for p in self.player_projectiles:
            p[1] -= self.PROJ_SPEED

        self.alien_projectiles = [p for p in self.alien_projectiles if p[1] < self.HEIGHT]
        for p in self.alien_projectiles:
            p[1] += self.PROJ_SPEED / 2

    def _handle_collisions(self):
        # Player projectiles vs Aliens
        for p_idx, p_pos in reversed(list(enumerate(self.player_projectiles))):
            for a_idx, alien in reversed(list(enumerate(self.aliens))):
                dist = np.linalg.norm(p_pos - alien['pos'])
                if dist < self.ALIEN_SIZE + self.PROJ_SIZE:
                    # sfx: explosion.wav
                    self._create_explosion(alien['pos'])
                    self.aliens.pop(a_idx)
                    self.player_projectiles.pop(p_idx)
                    self.score += 100
                    self.reward_this_step += 10
                    break
        
        # Alien projectiles vs Player
        for p_idx, p_pos in reversed(list(enumerate(self.alien_projectiles))):
            dist = np.linalg.norm(p_pos - self.player_pos)
            if dist < self.PLAYER_SIZE + self.PROJ_SIZE:
                # sfx: player_hit.wav
                self.alien_projectiles.pop(p_idx)
                self.player_health -= 1
                self.reward_this_step -= 10
                self._create_explosion(self.player_pos, is_player=True)
                if self.player_health <= 0:
                    self.game_over = True
                break
    
    def _update_waves(self):
        if not self.aliens and not self.game_over and not self.win_game:
            self.reward_this_step += 100 # Wave clear bonus
            self.current_wave += 1
            if self.current_wave >= self.WAVE_COUNT:
                self.win_game = True
                self.reward_this_step += 500 # Win game bonus
            else:
                self._spawn_wave()

    def _spawn_wave(self):
        rows = 2 + self.current_wave // 2
        cols = 6 + self.current_wave
        x_spacing = self.WIDTH * 0.8 / cols
        y_spacing = 30
        
        for r in range(rows):
            for c in range(cols):
                pos = np.array([
                    self.WIDTH * 0.1 + c * x_spacing + x_spacing / 2,
                    50 + r * y_spacing
                ], dtype=np.float32)
                self.aliens.append({'pos': pos})
        
        assert all(a['pos'][0] > 0 and a['pos'][0] < self.WIDTH for a in self.aliens), "Alien spawned off-screen"

    def _update_effects(self):
        # Explosions
        for exp in self.explosions:
            exp['life'] -= 1
        self.explosions = [exp for exp in self.explosions if exp['life'] > 0]
        
        # Stars
        for i, (x, y, speed) in enumerate(self.stars):
            y_new = y + speed
            if y_new > self.HEIGHT:
                y_new = 0
                x = self.np_random.integers(0, self.WIDTH)
            self.stars[i] = (x, y_new, speed)

    def _create_explosion(self, pos, is_player=False):
        num_particles = 20 if not is_player else 40
        for _ in range(num_particles):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 6)
            life = self.np_random.integers(15, 30)
            self.explosions.append({
                'pos': pos.copy(),
                'vel': np.array([math.cos(angle) * speed, math.sin(angle) * speed]),
                'life': life,
                'max_life': life
            })

    def _check_termination(self):
        if self.game_over or self.win_game:
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for x, y, speed in self.stars:
            size = int(speed)
            brightness = int(100 + speed * 50)
            color = (brightness, brightness, brightness)
            pygame.draw.rect(self.screen, color, (int(x), int(y), size, size))

    def _render_game(self):
        # Draw aliens
        for alien in self.aliens:
            pos = alien['pos'].astype(int)
            pts = [
                (pos[0], pos[1] + self.ALIEN_SIZE),
                (pos[0] - self.ALIEN_SIZE, pos[1] - self.ALIEN_SIZE),
                (pos[0] + self.ALIEN_SIZE, pos[1] - self.ALIEN_SIZE),
            ]
            pygame.gfxdraw.aapolygon(self.screen, pts, self.COLOR_ALIEN)
            pygame.gfxdraw.filled_polygon(self.screen, pts, self.COLOR_ALIEN)

        # Draw player projectiles
        for p in self.player_projectiles:
            pos = p.astype(int)
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ, (pos[0] - self.PROJ_SIZE, pos[1] - self.PROJ_SIZE * 2, self.PROJ_SIZE*2, self.PROJ_SIZE*4))

        # Draw alien projectiles
        for p in self.alien_projectiles:
            pos = p.astype(int)
            pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], self.PROJ_SIZE, self.COLOR_ALIEN_PROJ)
            pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], self.PROJ_SIZE, self.COLOR_ALIEN_PROJ)

        # Draw player
        if self.player_health > 0:
            pos = self.player_pos.astype(int)
            pts = [
                (pos[0], pos[1] - self.PLAYER_SIZE),
                (pos[0] - self.PLAYER_SIZE, pos[1] + self.PLAYER_SIZE),
                (pos[0] + self.PLAYER_SIZE, pos[1] + self.PLAYER_SIZE),
            ]
            pygame.gfxdraw.aapolygon(self.screen, pts, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, pts, self.COLOR_PLAYER)

        # Draw explosions
        for exp in self.explosions:
            exp['pos'] += exp['vel']
            progress = exp['life'] / exp['max_life']
            radius = int((1 - progress) * 15)
            color_idx = min(2, int((1 - progress) * 3))
            color = self.COLOR_EXPLOSION[color_idx]
            alpha_color = (*color, int(progress * 255))
            
            # Create a temporary surface for transparency
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, alpha_color, (radius, radius), radius)
            self.screen.blit(temp_surf, (int(exp['pos'][0] - radius), int(exp['pos'][1] - radius)))

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Wave
        wave_text = self.font_small.render(f"WAVE: {self.current_wave + 1}/{self.WAVE_COUNT}", True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))

        # Health bar
        bar_width = 200
        bar_height = 15
        bar_x = (self.WIDTH - bar_width) / 2
        bar_y = self.HEIGHT - bar_height - 10
        
        health_ratio = max(0, self.player_health / self.PLAYER_MAX_HEALTH)
        current_health_width = int(bar_width * health_ratio)

        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BORDER, (bar_x - 2, bar_y - 2, bar_width + 4, bar_height + 4))
        pygame.draw.rect(self.screen, self.COLOR_HEALTH_BG, (bar_x, bar_y, bar_width, bar_height))
        if current_health_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_FG, (bar_x, bar_y, current_health_width, bar_height))

        # Game Over / Win Text
        if self.game_over and self.player_health <= 0:
            text = self.font_large.render("GAME OVER", True, self.COLOR_ALIEN)
            text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text, text_rect)
        elif self.win_game:
            text = self.font_large.render("YOU WIN!", True, self.COLOR_PLAYER)
            text_rect = text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave + 1,
            "player_health": self.player_health,
            "aliens_remaining": len(self.aliens)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space
        self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        assert self.player_health == self.PLAYER_MAX_HEALTH
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == "__main__":
    # This block allows you to play the game directly
    import os
    os.environ["SDL_VIDEODRIVER"] = "x11" # Use "x11", "dummy", "windows", or "quartz"
    
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Alien Invasion")
    
    terminated = False
    total_reward = 0
    
    # Mapping from Pygame keys to MultiDiscrete actions
    key_map = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    while not terminated:
        # --- Human Input ---
        movement = 0
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        elif keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        # --- Environment Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # --- Rendering ---
        # The observation is already the rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward:.2f}")
    env.close()