
# Generated: 2025-08-28T05:20:11.356013
# Source Brief: brief_05536.md
# Brief Index: 5536

        
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
        "Controls: ←→ to move. Hold space to fire."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A top-down arcade shooter. Destroy waves of descending aliens to protect your ship."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 40, bold=True)
        
        # Colors
        self.COLOR_BG = (10, 5, 20)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_ENEMY = (255, 50, 50)
        self.COLOR_PLAYER_PROJ = (100, 255, 255)
        self.COLOR_ENEMY_PROJ = (255, 100, 255)
        self.COLOR_TEXT = (255, 255, 255)
        self.COLOR_EXPLOSION = [(255, 255, 0), (255, 128, 0), (255, 64, 0)]
        
        # Game constants
        self.MAX_STEPS = 10000
        self.MAX_WAVES = 10
        self.PLAYER_SPEED = 6
        self.PLAYER_FIRE_COOLDOWN = 6 # frames
        self.PROJECTILE_SPEED = 10
        
        # Initialize state variables
        self.player_pos = None
        self.player_lives = None
        self.player_fire_timer = None
        self.player_projectiles = None
        self.enemies = None
        self.enemy_projectiles = None
        self.enemy_move_dir = None
        self.enemy_move_down_timer = None
        self.particles = None
        self.stars = None
        self.current_wave = None
        self.steps = None
        self.score = None
        self.game_over = None
        self.game_won = None

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = pygame.Vector2(self.WIDTH / 2, self.HEIGHT - 40)
        self.player_lives = 3
        self.player_fire_timer = 0
        self.player_projectiles = []
        self.enemies = []
        self.enemy_projectiles = []
        self.enemy_move_dir = 1
        self.enemy_move_down_timer = 0
        self.particles = []
        
        self.current_wave = 1
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        if self.stars is None:
            self._spawn_stars()
            
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        self.steps += 1
        reward = 0.1 # Survival reward
        
        if not self.game_over:
            # Unpack factorized action
            movement = action[0]
            space_held = action[1] == 1
            
            # Update game logic
            reward += self._update_game_state(movement, space_held)
        
        terminated = self._check_termination()
        
        if self.game_over and not self.game_won:
            reward -= 100 # Penalty for losing
        elif self.game_won:
            reward += 1000 # Bonus for winning
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_game_state(self, movement, space_held):
        step_reward = 0
        
        # Handle player movement and firing
        if movement == 3: # Left
            self.player_pos.x -= self.PLAYER_SPEED
        elif movement == 4: # Right
            self.player_pos.x += self.PLAYER_SPEED
        self.player_pos.x = np.clip(self.player_pos.x, 20, self.WIDTH - 20)
        
        if self.player_fire_timer > 0:
            self.player_fire_timer -= 1
        
        if space_held and self.player_fire_timer == 0:
            # sfx: player_shoot.wav
            self.player_projectiles.append(pygame.Rect(self.player_pos.x - 2, self.player_pos.y - 20, 4, 15))
            self.player_fire_timer = self.PLAYER_FIRE_COOLDOWN

        # Update player projectiles
        missed_shots = 0
        for proj in self.player_projectiles[:]:
            proj.y -= self.PROJECTILE_SPEED
            if proj.bottom < 0:
                self.player_projectiles.remove(proj)
                missed_shots += 1
        step_reward -= missed_shots * 0.2

        # Update enemies
        self._update_enemies()
        
        # Update enemy projectiles
        for proj in self.enemy_projectiles[:]:
            proj.y += self.PROJECTILE_SPEED / 2
            if proj.top > self.HEIGHT:
                self.enemy_projectiles.remove(proj)

        # Update particles
        for p in self.particles[:]:
            p['pos'] += p['vel']
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
                
        # Handle collisions
        step_reward += self._handle_collisions()
        
        # Check for wave clear
        if not self.enemies and not self.game_over:
            step_reward += 100
            self.current_wave += 1
            if self.current_wave > self.MAX_WAVES:
                self.game_won = True
                self.game_over = True
            else:
                self._spawn_wave()
                
        return step_reward

    def _handle_collisions(self):
        collision_reward = 0
        # Player projectiles vs enemies
        for proj in self.player_projectiles[:]:
            for enemy_rect in self.enemies[:]:
                if proj.colliderect(enemy_rect):
                    # sfx: explosion.wav
                    self._create_explosion(enemy_rect.center, self.COLOR_EXPLOSION)
                    self.enemies.remove(enemy_rect)
                    if proj in self.player_projectiles: self.player_projectiles.remove(proj)
                    self.score += 10
                    collision_reward += 10
                    break

        # Enemy projectiles vs player
        player_rect = pygame.Rect(self.player_pos.x - 15, self.player_pos.y - 10, 30, 20)
        for proj in self.enemy_projectiles[:]:
            if player_rect.colliderect(proj):
                self.enemy_projectiles.remove(proj)
                # sfx: player_hit.wav
                self._create_explosion(self.player_pos, self.COLOR_EXPLOSION)
                self.player_lives -= 1
                collision_reward -= 50 # Penalty for getting hit
                if self.player_lives <= 0:
                    self.game_over = True
                break
        return collision_reward

    def _update_enemies(self):
        if not self.enemies:
            return

        # Difficulty scaling
        speed_multiplier = 1 + (self.current_wave - 1) * 0.1
        enemy_speed = 1.0 * speed_multiplier
        fire_rate_multiplier = 1 + (self.current_wave - 1) * 0.1
        enemy_fire_rate = 0.01 * fire_rate_multiplier

        move_down = False
        if self.enemy_move_down_timer > 0:
            self.enemy_move_down_timer -= 1
            for enemy in self.enemies:
                enemy.y += 1
        else:
            for enemy in self.enemies:
                enemy.x += self.enemy_move_dir * enemy_speed
                if enemy.right > self.WIDTH or enemy.left < 0:
                    move_down = True
        
        if move_down:
            self.enemy_move_dir *= -1
            self.enemy_move_down_timer = 10

        # Enemy firing and descent
        for enemy in self.enemies:
            if enemy.bottom > self.HEIGHT - 60: # Aliens reached defensive line
                self.game_over = True
                self.player_lives = 0
                return
            if self.np_random.random() < enemy_fire_rate:
                # sfx: enemy_shoot.wav
                self.enemy_projectiles.append(pygame.Rect(enemy.centerx - 2, enemy.centery, 4, 10))

    def _spawn_wave(self):
        rows, cols = 4, 8
        for r in range(rows):
            for c in range(cols):
                x = c * 60 + (self.WIDTH - cols * 60) / 2 + 30
                y = r * 40 + 50
                self.enemies.append(pygame.Rect(x, y, 30, 20))
                
    def _spawn_stars(self):
        self.stars = []
        for _ in range(150):
            self.stars.append({
                'pos': pygame.Vector2(self.np_random.integers(0, self.WIDTH), self.np_random.integers(0, self.HEIGHT)),
                'size': self.np_random.choice([1, 2, 3]),
                'speed': self.np_random.uniform(0.2, 0.8)
            })

    def _check_termination(self):
        if self.game_over:
            return True
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            return True
        return False

    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stars
        for star in self.stars:
            star['pos'].y += star['speed']
            if star['pos'].y > self.HEIGHT:
                star['pos'].y = 0
                star['pos'].x = self.np_random.integers(0, self.WIDTH)
            brightness = int(100 + star['speed'] * 100)
            pygame.draw.circle(self.screen, (brightness, brightness, brightness), (int(star['pos'].x), int(star['pos'].y)), star['size'] // 2)

        # Player projectiles
        for proj in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ, proj)
            pygame.gfxdraw.rectangle(self.screen, proj, (*self.COLOR_PLAYER_PROJ, 150))

        # Enemy projectiles
        for proj in self.enemy_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_ENEMY_PROJ, proj)
            
        # Enemies
        for enemy_rect in self.enemies:
            pygame.draw.rect(self.screen, self.COLOR_ENEMY, enemy_rect, border_radius=3)
            
        # Player
        if self.player_lives > 0:
            p = self.player_pos
            points = [(p.x, p.y - 15), (p.x - 15, p.y + 10), (p.x + 15, p.y + 10)]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Particles (explosions)
        for p in self.particles:
            life_ratio = p['life'] / p['max_life']
            size = int(p['size'] * life_ratio)
            if size > 0:
                alpha = int(255 * life_ratio)
                color = (p['color'][0], p['color'][1], p['color'][2], alpha)
                temp_surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
                pygame.draw.circle(temp_surf, color, (size, size), size)
                self.screen.blit(temp_surf, (p['pos'].x - size, p['pos'].y - size), special_flags=pygame.BLEND_RGBA_ADD)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Wave
        wave_text = self.font_ui.render(f"WAVE: {self.current_wave}/{self.MAX_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, (self.WIDTH - wave_text.get_width() - 10, 10))
        
        # Lives
        for i in range(self.player_lives):
            points = [(25 + i * 35, self.HEIGHT - 25), (10 + i * 35, self.HEIGHT - 10), (40 + i * 35, self.HEIGHT - 10)]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)
            
        # Game Over / Win message
        if self.game_over:
            msg = "GAME OVER" if not self.game_won else "YOU WIN!"
            color = self.COLOR_ENEMY if not self.game_won else self.COLOR_PLAYER
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)
            
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "wave": self.current_wave
        }

    def _create_explosion(self, pos, colors):
        for _ in range(25):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 5)
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': pygame.Vector2(pos),
                'vel': pygame.Vector2(math.cos(angle) * speed, math.sin(angle) * speed),
                'life': life,
                'max_life': life,
                'size': self.np_random.integers(5, 12),
                'color': random.choice(colors)
            })
            
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