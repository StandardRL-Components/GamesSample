
# Generated: 2025-08-28T04:27:06.896663
# Source Brief: brief_02317.md
# Brief Index: 2317

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    user_guide = "Controls: Use ↑ and ↓ to move your ship. Press Space to fire your laser."
    game_description = "Pilot your ship and blast waves of alien invaders in this retro arcade shooter. Clear all 50 aliens to win!"
    
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    
    # Colors
    COLOR_BG = (10, 5, 20)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_PLAYER_GLOW = (150, 255, 150)
    COLOR_ALIEN = (255, 50, 50)
    COLOR_PLAYER_BULLET = (255, 255, 100)
    COLOR_ALIEN_BULLET = (255, 100, 255)
    COLOR_EXPLOSION = (255, 255, 255)
    COLOR_TEXT = (220, 220, 220)
    STAR_COLORS = [(50, 50, 70), (100, 100, 120), (200, 200, 220)]

    # Game parameters
    PLAYER_SPEED = 8
    PLAYER_FIRE_COOLDOWN = 6  # 5 shots per second at 30fps
    BULLET_SPEED = 15
    ALIEN_SPEED_X = 2
    MAX_STEPS = 5000
    PLAYER_LIVES = 3
    TOTAL_ALIENS = 50
    INVINCIBILITY_FRAMES = 90 # 3 seconds at 30fps

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.font_ui = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # State variables initialized in reset()
        self.player = None
        self.aliens = []
        self.player_bullets = []
        self.alien_bullets = []
        self.explosions = []
        self.stars = []
        self.steps = 0
        self.score = 0
        self.lives = 0
        self.aliens_destroyed = 0
        self.last_kill_step = -100
        self.game_over = False
        self.game_won = False
        self.invincibility_timer = 0
        self.alien_fire_prob = 0.0

        self.reset()
        
        # This check is useful for development
        # self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.lives = self.PLAYER_LIVES
        self.aliens_destroyed = 0
        self.last_kill_step = -100
        self.game_over = False
        self.game_won = False
        self.invincibility_timer = 0
        self.alien_fire_prob = 0.005

        self.player = {
            "rect": pygame.Rect(50, self.SCREEN_HEIGHT // 2 - 15, 30, 30),
            "fire_cooldown": 0,
        }
        
        self.aliens.clear()
        self.player_bullets.clear()
        self.alien_bullets.clear()
        self.explosions.clear()
        
        self._create_stars()
        self._create_aliens()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            # If the game is over, do nothing but return the final state
            reward = 0
            terminated = True
            return self._get_observation(), reward, terminated, False, self._get_info()
            
        reward = 0.1  # Survival reward

        # --- 1. Handle Input & Player Updates ---
        self._handle_input(action)
        self.player["fire_cooldown"] = max(0, self.player["fire_cooldown"] - 1)
        self.invincibility_timer = max(0, self.invincibility_timer - 1)

        # --- 2. Update Game Objects ---
        reward += self._update_player_bullets()
        self._update_aliens()
        self._update_alien_bullets()
        self._update_explosions()
        self._update_stars()

        # --- 3. Collision Detection ---
        reward += self._check_collisions()

        # --- 4. Check Termination Conditions ---
        self.steps += 1
        terminated = False
        terminal_reward = 0
        
        if self.lives <= 0:
            terminated = True
            terminal_reward = -100
            self.game_over = True
        elif self.aliens_destroyed >= self.TOTAL_ALIENS:
            terminated = True
            terminal_reward = 100
            self.game_over = True
            self.game_won = True
        elif self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        reward += terminal_reward
        self.score += max(0, reward) # Score doesn't decrease

        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _handle_input(self, action):
        movement = action[0]
        space_held = action[1] == 1
        
        # Player movement
        if movement == 1:  # Up
            self.player["rect"].y -= self.PLAYER_SPEED
        elif movement == 2:  # Down
            self.player["rect"].y += self.PLAYER_SPEED
        
        # Clamp player position
        self.player["rect"].y = np.clip(self.player["rect"].y, 0, self.SCREEN_HEIGHT - self.player["rect"].height)

        # Player firing
        if space_held and self.player["fire_cooldown"] == 0:
            # sfx: player_shoot.wav
            bullet_rect = pygame.Rect(self.player["rect"].right, self.player["rect"].centery - 2, 12, 4)
            self.player_bullets.append(bullet_rect)
            self.player["fire_cooldown"] = self.PLAYER_FIRE_COOLDOWN

    def _update_player_bullets(self):
        reward = 0
        for bullet in self.player_bullets[:]:
            bullet.x += self.BULLET_SPEED
            if bullet.left > self.SCREEN_WIDTH:
                self.player_bullets.remove(bullet)
                reward -= 0.2  # Penalty for missing
        return reward

    def _update_aliens(self):
        for alien in self.aliens:
            alien["x"] -= self.ALIEN_SPEED_X
            alien["rect"].x = int(alien["x"])
            alien["rect"].y = int(alien["y_base"] + math.sin(alien["x"] * alien["freq"] + alien["phase"]) * alien["amp"])
            
            # Alien firing logic
            if alien["rect"].x < self.SCREEN_WIDTH and self.np_random.random() < self.alien_fire_prob:
                # sfx: alien_shoot.wav
                bullet_rect = pygame.Rect(alien["rect"].left - 10, alien["rect"].centery - 2, 10, 4)
                self.alien_bullets.append(bullet_rect)

    def _update_alien_bullets(self):
        for bullet in self.alien_bullets[:]:
            bullet.x -= self.BULLET_SPEED
            if bullet.right < 0:
                self.alien_bullets.remove(bullet)

    def _update_explosions(self):
        for explosion in self.explosions[:]:
            explosion["life"] -= 1
            if explosion["life"] <= 0:
                self.explosions.remove(explosion)

    def _update_stars(self):
        for star in self.stars:
            star["pos"][0] -= star["speed"]
            if star["pos"][0] < 0:
                star["pos"][0] = self.SCREEN_WIDTH
                star["pos"][1] = self.np_random.integers(0, self.SCREEN_HEIGHT)

    def _check_collisions(self):
        reward = 0
        
        # Player bullets vs Aliens
        for bullet in self.player_bullets[:]:
            collided_alien = None
            for alien in self.aliens:
                if alien["rect"].colliderect(bullet):
                    collided_alien = alien
                    break
            
            if collided_alien:
                # sfx: explosion.wav
                self.player_bullets.remove(bullet)
                self.aliens.remove(collided_alien)
                self._create_explosion(collided_alien["rect"].center, 30)
                
                reward += 1.0  # Reward for hit
                if self.steps - self.last_kill_step <= 2:
                    reward += 2.0  # Chain kill bonus
                
                self.last_kill_step = self.steps
                self.aliens_destroyed += 1

                # Increase difficulty
                if self.aliens_destroyed % 10 == 0:
                    self.alien_fire_prob += 0.01

        # Alien bullets vs Player
        if self.invincibility_timer == 0:
            for bullet in self.alien_bullets[:]:
                if self.player["rect"].colliderect(bullet):
                    # sfx: player_hit.wav
                    self.alien_bullets.remove(bullet)
                    self.lives -= 1
                    reward -= 1.0  # Penalty for being hit
                    self.invincibility_timer = self.INVINCIBILITY_FRAMES
                    self._create_explosion(self.player["rect"].center, 40)
                    break
        
        return reward

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        for star in self.stars:
            pygame.draw.circle(self.screen, star["color"], (int(star["pos"][0]), int(star["pos"][1])), star["radius"])

    def _render_game(self):
        # Draw aliens
        for alien in self.aliens:
            if alien["rect"].right > 0:
                 pygame.draw.rect(self.screen, self.COLOR_ALIEN, alien["rect"])
                 pygame.draw.rect(self.screen, (255,150,150), alien["rect"], 1)

        # Draw player bullets
        for bullet in self.player_bullets:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_BULLET, bullet)

        # Draw alien bullets
        for bullet in self.alien_bullets:
            pygame.draw.rect(self.screen, self.COLOR_ALIEN_BULLET, bullet)
            
        # Draw player (with invincibility flash)
        if self.invincibility_timer == 0 or self.invincibility_timer % 10 < 5:
            # Draw glow
            glow_rect = self.player["rect"].inflate(10, 10)
            s = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.circle(s, (*self.COLOR_PLAYER_GLOW, 50), (glow_rect.width//2, glow_rect.height//2), glow_rect.width//2)
            self.screen.blit(s, glow_rect.topleft)
            
            # Draw ship
            p = self.player["rect"]
            points = [(p.right, p.top), (p.right, p.bottom), (p.left, p.centery)]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)
            pygame.draw.aalines(self.screen, self.COLOR_PLAYER_GLOW, True, points)

        # Draw explosions
        for explosion in self.explosions:
            progress = (explosion["max_life"] - explosion["life"]) / explosion["max_life"]
            radius = int(explosion["radius"] * progress)
            alpha = int(255 * (1 - progress))
            pygame.gfxdraw.filled_circle(self.screen, int(explosion["pos"][0]), int(explosion["pos"][1]), radius, (*self.COLOR_EXPLOSION, alpha))

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Lives
        lives_text = self.font_ui.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (self.SCREEN_WIDTH - 150, 10))
        for i in range(self.lives):
            ship_rect = pygame.Rect(self.SCREEN_WIDTH - 70 + (i * 20), 12, 15, 15)
            points = [(ship_rect.right, ship_rect.top), (ship_rect.right, ship_rect.bottom), (ship_rect.left, ship_rect.centery)]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)

        # Game Over / You Win
        if self.game_over:
            msg = "GAME OVER" if not self.game_won else "YOU WIN!"
            end_text = self.font_game_over.render(msg, True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.lives,
            "aliens_destroyed": self.aliens_destroyed,
        }
    
    def _create_stars(self):
        self.stars.clear()
        for _ in range(150):
            layer = self.np_random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
            self.stars.append({
                "pos": [self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT)],
                "speed": [0.5, 1, 1.5][layer],
                "radius": [1, 1, 2][layer],
                "color": self.STAR_COLORS[layer],
            })

    def _create_aliens(self):
        self.aliens.clear()
        rows, cols = 5, 10
        for r in range(rows):
            for c in range(cols):
                self.aliens.append({
                    "rect": pygame.Rect(0, 0, 25, 20),
                    "x": self.SCREEN_WIDTH + 50 + c * 50,
                    "y_base": 60 + r * 50,
                    "freq": self.np_random.uniform(0.01, 0.03),
                    "amp": self.np_random.uniform(10, 30),
                    "phase": self.np_random.uniform(0, 2 * math.pi)
                })

    def _create_explosion(self, pos, radius):
        self.explosions.append({
            "pos": pos,
            "radius": radius,
            "life": 15,
            "max_life": 15
        })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation.
        '''
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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("Galactic Defender")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        # --- Human Controls ---
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            movement = 1
        elif keys[pygame.K_DOWN]:
            movement = 2
        
        if keys[pygame.K_SPACE]:
            space_held = 1

        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            
        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit to 30 FPS for interactive play

    env.close()