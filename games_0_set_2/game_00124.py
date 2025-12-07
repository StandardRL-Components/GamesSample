
# Generated: 2025-08-27T12:39:50.902610
# Source Brief: brief_00124.md
# Brief Index: 124

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move your ship. Press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A retro top-down shooter. Survive waves of descending aliens and achieve a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and world dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.MAX_STEPS = 10000
        self.FPS = 30

        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_ui = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 72)

        # Colors
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_PLAYER = (0, 255, 128)
        self.COLOR_PLAYER_PROJ = (255, 255, 0)
        self.COLOR_ALIEN_PROJ = (255, 255, 255)
        self.COLOR_TEXT = (220, 220, 220)
        self.WAVE_COLORS = [
            (255, 50, 50),   # Red
            (50, 150, 255),  # Blue
            (200, 50, 255),  # Purple
            (255, 150, 50),  # Orange
            (50, 255, 50),   # Green
        ]

        # Game parameters
        self.PLAYER_SPEED = 8
        self.PLAYER_FIRE_COOLDOWN = 6 # frames
        self.PLAYER_PROJ_SPEED = 12
        self.ALIEN_PROJ_SPEED = 5
        self.MAX_WAVES = 5

        # Initialize state variables
        self.player_pos = pygame.Vector2(0, 0)
        self.player_lives = 0
        self.player_projectiles = []
        self.player_fire_timer = 0
        self.aliens = []
        self.alien_projectiles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.wave = 0
        self.game_over = False
        self.game_won = False
        self.rng = np.random.default_rng()
        
        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)

        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 40)
        self.player_lives = 3
        self.player_projectiles.clear()
        self.player_fire_timer = 0
        self.aliens.clear()
        self.alien_projectiles.clear()
        self.particles.clear()
        
        self.steps = 0
        self.score = 0
        self.wave = 1
        self.game_over = False
        self.game_won = False
        
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0.1  # Survival reward
        
        # --- Handle Input and Player Movement ---
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        self._handle_player_input(movement, space_held)

        # --- Update Game Objects ---
        self._update_projectiles()
        self._update_aliens()
        self._update_particles()
        
        # --- Collision Detection ---
        collision_reward = self._handle_collisions()
        reward += collision_reward

        # --- Check Game State ---
        if not self.aliens and not self.game_won:
            if self.wave < self.MAX_WAVES:
                self.wave += 1
                self._spawn_wave()
            else:
                self.game_won = True
                self.game_over = True
                reward += 100 # Win bonus
        
        if self.player_lives <= 0 and not self.game_over:
            self.game_over = True
            reward -= 100 # Loss penalty

        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            self.game_over,
            False,
            self._get_info()
        )

    def _spawn_wave(self):
        num_aliens = 8 + self.wave * 4
        rows = 2 + math.floor(self.wave / 2)
        cols = math.ceil(num_aliens / rows)
        x_spacing = self.WIDTH * 0.8 / max(1, cols - 1) if cols > 1 else 0
        y_spacing = 40

        for i in range(num_aliens):
            row = i // cols
            col = i % cols
            x = self.WIDTH * 0.1 + col * x_spacing
            y = 50 + row * y_spacing
            
            self.aliens.append({
                "pos": pygame.Vector2(x, y),
                "dir": 1 if col % 2 == 0 else -1
            })

    def _handle_player_input(self, movement, space_held):
        # Movement
        if movement == 1: self.player_pos.y -= self.PLAYER_SPEED
        if movement == 2: self.player_pos.y += self.PLAYER_SPEED
        if movement == 3: self.player_pos.x -= self.PLAYER_SPEED
        if movement == 4: self.player_pos.x += self.PLAYER_SPEED
        
        # Clamp player position to screen bounds
        self.player_pos.x = np.clip(self.player_pos.x, 15, self.WIDTH - 15)
        self.player_pos.y = np.clip(self.player_pos.y, 15, self.HEIGHT - 15)

        # Firing
        if self.player_fire_timer > 0:
            self.player_fire_timer -= 1
        
        if space_held and self.player_fire_timer == 0:
            # sfx: player_shoot
            self.player_projectiles.append(self.player_pos.copy())
            self.player_fire_timer = self.PLAYER_FIRE_COOLDOWN

    def _update_projectiles(self):
        # Player projectiles
        for proj in self.player_projectiles[:]:
            proj.y -= self.PLAYER_PROJ_SPEED
            if proj.y < 0:
                self.player_projectiles.remove(proj)
        
        # Alien projectiles
        for proj in self.alien_projectiles[:]:
            proj.y += self.ALIEN_PROJ_SPEED
            if proj.y > self.HEIGHT:
                self.alien_projectiles.remove(proj)

    def _update_aliens(self):
        base_speed_x = 0.5 + self.wave * 0.2
        base_speed_y = 0.2 + self.wave * 0.05
        fire_prob = 0.001 + self.wave * 0.001

        for alien in self.aliens:
            alien['pos'].x += alien['dir'] * base_speed_x
            alien['pos'].y += base_speed_y
            
            if alien['pos'].x < 20 or alien['pos'].x > self.WIDTH - 20:
                alien['dir'] *= -1

            if self.rng.random() < fire_prob:
                # sfx: alien_shoot
                self.alien_projectiles.append(alien['pos'].copy())

    def _update_particles(self):
        for p in self.particles[:]:
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                p['pos'] += p['vel']
                p['radius'] -= p['decay']

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs Aliens
        for proj in self.player_projectiles[:]:
            for alien in self.aliens[:]:
                if proj.distance_to(alien['pos']) < 15: # Collision radius
                    # sfx: alien_explode
                    self._create_explosion(alien['pos'], self.WAVE_COLORS[self.wave - 1], 20)
                    self.aliens.remove(alien)
                    if proj in self.player_projectiles:
                        self.player_projectiles.remove(proj)
                    self.score += 10
                    reward += 1
                    break

        # Alien projectiles vs Player
        for proj in self.alien_projectiles[:]:
            if proj.distance_to(self.player_pos) < 15:
                self.alien_projectiles.remove(proj)
                self._lose_life()
                reward -= 10
                break

        # Aliens vs Player
        for alien in self.aliens[:]:
            if alien['pos'].distance_to(self.player_pos) < 20:
                self.aliens.remove(alien)
                self._lose_life()
                reward -= 10
                break
        
        return reward

    def _lose_life(self):
        # sfx: player_explode
        self.player_lives -= 1
        self._create_explosion(self.player_pos, self.COLOR_PLAYER, 30)
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 40)
        if self.player_lives <= 0:
            self.game_over = True

    def _create_explosion(self, pos, color, num_particles):
        for _ in range(num_particles):
            angle = self.rng.random() * 2 * math.pi
            speed = 1 + self.rng.random() * 3
            vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
            self.particles.append({
                'pos': pos.copy(),
                'vel': vel,
                'life': 10 + self.rng.integers(0, 10),
                'radius': 5 + self.rng.random() * 3,
                'color': color,
                'decay': 0.2
            })
    
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _render_game(self):
        # Render particles
        for p in self.particles:
            if p['radius'] > 0:
                pygame.draw.circle(self.screen, p['color'], p['pos'], max(0, int(p['radius'])))

        # Render alien projectiles
        for proj in self.alien_projectiles:
            pygame.draw.circle(self.screen, self.COLOR_ALIEN_PROJ, (int(proj.x), int(proj.y)), 3)

        # Render player projectiles
        for proj in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ, (int(proj.x - 2), int(proj.y - 8), 4, 16))
            
        # Render aliens
        alien_color = self.WAVE_COLORS[self.wave - 1]
        for alien in self.aliens:
            pos = (int(alien['pos'].x), int(alien['pos'].y))
            pygame.draw.rect(self.screen, alien_color, (pos[0] - 8, pos[1] - 8, 16, 16))

        # Render player
        if self.player_lives > 0:
            p = self.player_pos
            points = [
                (p.x, p.y - 12),
                (p.x - 10, p.y + 10),
                (p.x + 10, p.y + 10)
            ]
            pygame.gfxdraw.aapolygon(self.screen, [(int(x), int(y)) for x, y in points], self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, [(int(x), int(y)) for x, y in points], self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Wave
        wave_text = self.font_ui.render(f"WAVE: {self.wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))

        # Lives
        for i in range(self.player_lives):
            p = pygame.Vector2(25 + i * 25, self.HEIGHT - 20)
            points = [
                (p.x, p.y - 8),
                (p.x - 6, p.y + 6),
                (p.x + 6, p.y + 6)
            ]
            pygame.gfxdraw.filled_polygon(self.screen, [(int(x), int(y)) for x, y in points], self.COLOR_PLAYER)
            
        # Game Over / Win Message
        if self.game_over:
            message = "YOU WIN!" if self.game_won else "GAME OVER"
            color = self.COLOR_PLAYER if self.game_won else self.WAVE_COLORS[0]
            end_text = self.font_title.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.wave,
            "lives": self.player_lives,
        }

    def close(self):
        pygame.quit()

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
    # To play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    
    terminated = False
    
    while not terminated:
        # --- Human Input ---
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]

        # --- Gym Step ---
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- Pygame Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # --- Event Handling & Clock ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
        
        clock.tick(env.FPS)

        if terminated:
            print(f"Game Over. Final Info: {info}")
            # Wait a bit before closing
            pygame.time.wait(2000)

    env.close()