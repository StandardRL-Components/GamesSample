
# Generated: 2025-08-27T19:03:02.931938
# Source Brief: brief_02036.md
# Brief Index: 2036

        
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
        "Controls: Arrow keys to move. Hold Space to fire. Press Shift to activate shield (5s cooldown)."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Top-down space shooter. Destroy all alien invaders across 3 stages while dodging their fire. You have 3 lives and a 60-second timer per stage."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.FPS = 30
        self.MAX_STAGES = 3
        self.STAGE_TIME_LIMIT = 60 * self.FPS

        # Colors
        self.COLOR_BG = (10, 5, 25)
        self.COLOR_STAR = (200, 200, 200)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_SHIELD = (100, 150, 255, 100)
        self.COLOR_ALIEN = (255, 50, 50)
        self.COLOR_PLAYER_PROJ = (100, 150, 255)
        self.COLOR_ALIEN_PROJ = (255, 255, 0)
        self.COLOR_EXPLOSION = (255, 120, 0)
        self.COLOR_UI_TEXT = (220, 220, 220)
        self.COLOR_UI_BAR_BG = (50, 50, 50)
        self.COLOR_UI_BAR_FG = (100, 150, 255)

        # Player properties
        self.PLAYER_SIZE = 20
        self.PLAYER_SPEED = 6
        self.PLAYER_FIRE_COOLDOWN_MAX = 8  # ~4 shots/sec
        self.PLAYER_SHIELD_COOLDOWN_MAX = 5 * self.FPS # 5 seconds

        # Alien properties
        self.ALIEN_SIZE = 20
        self.ALIENS_PER_STAGE = 25

        # Projectile properties
        self.PROJ_SIZE = 4
        self.PLAYER_PROJ_SPEED = 10
        self.ALIEN_PROJ_SPEED = 4

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
        self.font_small = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 48)

        # Initialize state variables
        self.player_pos = None
        self.player_lives = None
        self.player_fire_cooldown = None
        self.player_shield_active = None
        self.player_shield_cooldown = None
        self.aliens = None
        self.player_projectiles = None
        self.alien_projectiles = None
        self.explosions = None
        self.stars = None
        self.current_stage = None
        self.stage_timer = None
        self.score = None
        self.steps = None
        self.game_over = None
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_stage = 1
        
        self.player_pos = [self.WIDTH / 2, self.HEIGHT - 40]
        self.player_lives = 3
        self.player_fire_cooldown = 0
        self.player_shield_active = False
        self.player_shield_cooldown = 0

        self.player_projectiles = []
        self.alien_projectiles = []
        self.explosions = []
        
        self.stars = [
            (random.randint(0, self.WIDTH), random.randint(0, self.HEIGHT), random.randint(1, 2))
            for _ in range(100)
        ]

        self._reset_stage()

        return self._get_observation(), self._get_info()
    
    def _reset_stage(self):
        self.stage_timer = self.STAGE_TIME_LIMIT
        self.player_projectiles.clear()
        self.alien_projectiles.clear()
        self.explosions.clear()
        self._spawn_aliens()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        reward = 0
        self.steps += 1
        
        # Unpack action
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        # --- Handle Input & Player Actions ---
        if movement != 0:
            reward -= 0.001 # Small penalty for movement to encourage efficiency

        # Movement
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED
        if movement == 2: self.player_pos[1] += self.PLAYER_SPEED
        if movement == 3: self.player_pos[0] -= self.PLAYER_SPEED
        if movement == 4: self.player_pos[0] += self.PLAYER_SPEED
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE/2, self.WIDTH - self.PLAYER_SIZE/2)
        self.player_pos[1] = np.clip(self.player_pos[1], self.PLAYER_SIZE/2, self.HEIGHT - self.PLAYER_SIZE/2)

        # Firing
        if space_held and self.player_fire_cooldown == 0:
            # sfx: player_shoot.wav
            proj_pos = [self.player_pos[0], self.player_pos[1] - self.PLAYER_SIZE / 2]
            self.player_projectiles.append(pygame.Rect(proj_pos[0] - self.PROJ_SIZE/2, proj_pos[1], self.PROJ_SIZE, self.PROJ_SIZE*2))
            self.player_fire_cooldown = self.PLAYER_FIRE_COOLDOWN_MAX

        # Shield
        if shift_held and self.player_shield_cooldown == 0 and not self.player_shield_active:
            # sfx: shield_activate.wav
            self.player_shield_active = True
            self.player_shield_cooldown = self.PLAYER_SHIELD_COOLDOWN_MAX

        # --- Update Game State ---
        self._update_cooldowns()
        self._update_projectiles()
        reward += self._update_aliens()
        self._update_explosions()

        # --- Collisions ---
        reward += self._handle_collisions()

        # --- Check Game Flow ---
        # Stage Timer
        self.stage_timer -= 1
        if self.stage_timer <= 0:
            self.player_lives -= 1
            reward -= 50 # Penalty for timeout
            # sfx: life_lost.wav
            if self.player_lives > 0:
                self._reset_stage()
            else:
                self.game_over = True
        
        # Stage Completion
        if not self.aliens and self.current_stage <= self.MAX_STAGES:
            reward += 100 # Stage clear bonus
            self.current_stage += 1
            if self.current_stage > self.MAX_STAGES:
                self.game_over = True # Win condition
                reward += 300 # Win bonus
            else:
                # sfx: stage_clear.wav
                self._reset_stage()

        # Termination
        if self.player_lives <= 0:
            self.game_over = True
            reward -= 100 # Game over penalty
        
        terminated = self.game_over
        self.score += reward

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _update_cooldowns(self):
        if self.player_fire_cooldown > 0: self.player_fire_cooldown -= 1
        if self.player_shield_cooldown > 0: self.player_shield_cooldown -= 1

    def _update_projectiles(self):
        # Player projectiles
        for proj in self.player_projectiles[:]:
            proj.y -= self.PLAYER_PROJ_SPEED
            if proj.bottom < 0:
                self.player_projectiles.remove(proj)
        
        # Alien projectiles
        for proj_data in self.alien_projectiles[:]:
            proj_data['rect'].y += proj_data['speed']
            if proj_data['rect'].top > self.HEIGHT:
                self.alien_projectiles.remove(proj_data)
            # Check for dodge
            elif not proj_data['dodged'] and proj_data['rect'].top > self.player_pos[1]:
                self.score += 0.1 # Reward for dodging
                proj_data['dodged'] = True

    def _update_aliens(self):
        difficulty_mod = 1 + (self.current_stage - 1) * 0.1
        reward = 0
        for alien in self.aliens:
            # Movement
            alien['pos'][0] += alien['vel'][0] * difficulty_mod
            if alien['pos'][0] < self.ALIEN_SIZE/2 or alien['pos'][0] > self.WIDTH - self.ALIEN_SIZE/2:
                alien['vel'][0] *= -1
            alien['rect'].center = alien['pos']

            # Firing
            alien['fire_cooldown'] -= 1
            if alien['fire_cooldown'] <= 0:
                # sfx: alien_shoot.wav
                proj_speed = self.ALIEN_PROJ_SPEED * difficulty_mod
                proj_rect = pygame.Rect(alien['rect'].centerx - self.PROJ_SIZE/2, alien['rect'].bottom, self.PROJ_SIZE, self.PROJ_SIZE*2)
                self.alien_projectiles.append({'rect': proj_rect, 'speed': proj_speed, 'dodged': False})
                alien['fire_cooldown'] = random.randint(int(60 / difficulty_mod), int(120 / difficulty_mod))
        return reward
    
    def _update_explosions(self):
        for exp in self.explosions[:]:
            exp['life'] -= 1
            exp['radius'] += 2
            if exp['life'] <= 0:
                self.explosions.remove(exp)

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_SIZE/2, self.player_pos[1] - self.PLAYER_SIZE/2, self.PLAYER_SIZE, self.PLAYER_SIZE)

        # Player projectiles vs Aliens
        for proj in self.player_projectiles[:]:
            for alien in self.aliens[:]:
                if proj.colliderect(alien['rect']):
                    # sfx: explosion.wav
                    self.explosions.append({'pos': alien['rect'].center, 'radius': 0, 'life': 15, 'max_radius': 30})
                    self.aliens.remove(alien)
                    if proj in self.player_projectiles: self.player_projectiles.remove(proj)
                    reward += 1 # Reward for destroying alien
                    break
        
        # Alien projectiles vs Player
        for proj_data in self.alien_projectiles[:]:
            if player_rect.colliderect(proj_data['rect']):
                if self.player_shield_active:
                    # sfx: shield_hit.wav
                    self.player_shield_active = False
                else:
                    # sfx: player_hit.wav
                    self.player_lives -= 1
                    self.explosions.append({'pos': self.player_pos, 'radius': 0, 'life': 20, 'max_radius': 40})
                self.alien_projectiles.remove(proj_data)
        
        return reward

    def _spawn_aliens(self):
        self.aliens = []
        rows, cols = 5, 5
        x_spacing = self.WIDTH / (cols + 1)
        y_spacing = 50
        difficulty_mod = 1 + (self.current_stage - 1) * 0.1
        
        for r in range(rows):
            for c in range(cols):
                x = (c + 1) * x_spacing
                y = (r + 1) * y_spacing
                alien_rect = pygame.Rect(x - self.ALIEN_SIZE/2, y - self.ALIEN_SIZE/2, self.ALIEN_SIZE, self.ALIEN_SIZE)
                self.aliens.append({
                    'rect': alien_rect,
                    'pos': [x, y],
                    'vel': [random.choice([-1, 1]) * (1 + random.random()), 0],
                    'fire_cooldown': random.randint(30, int(120 / difficulty_mod))
                })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Stars
        for x, y, size in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STAR, (x, y), size)
        
        # Explosions
        for exp in self.explosions:
            alpha = int(255 * (exp['life'] / 15))
            pygame.gfxdraw.filled_circle(self.screen, int(exp['pos'][0]), int(exp['pos'][1]), int(exp['radius']), (*self.COLOR_EXPLOSION, alpha))
            pygame.gfxdraw.aacircle(self.screen, int(exp['pos'][0]), int(exp['pos'][1]), int(exp['radius']), (*self.COLOR_EXPLOSION, alpha))

        # Aliens
        for alien in self.aliens:
            pygame.draw.rect(self.screen, self.COLOR_ALIEN, alien['rect'])
        
        # Projectiles
        for proj in self.player_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ, proj)
        for proj_data in self.alien_projectiles:
            pygame.draw.rect(self.screen, self.COLOR_ALIEN_PROJ, proj_data['rect'])

        # Player
        player_points = [
            (self.player_pos[0], self.player_pos[1] - self.PLAYER_SIZE / 2),
            (self.player_pos[0] - self.PLAYER_SIZE / 2, self.player_pos[1] + self.PLAYER_SIZE / 2),
            (self.player_pos[0] + self.PLAYER_SIZE / 2, self.player_pos[1] + self.PLAYER_SIZE / 2),
        ]
        pygame.draw.polygon(self.screen, self.COLOR_PLAYER, [(int(p[0]), int(p[1])) for p in player_points])
        
        # Shield
        if self.player_shield_active:
            pygame.gfxdraw.filled_circle(self.screen, int(self.player_pos[0]), int(self.player_pos[1]), int(self.PLAYER_SIZE * 0.8), self.COLOR_SHIELD)
            pygame.gfxdraw.aacircle(self.screen, int(self.player_pos[0]), int(self.player_pos[1]), int(self.PLAYER_SIZE * 0.8), self.COLOR_SHIELD)

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Stage and Timer
        stage_text = self.font_small.render(f"STAGE: {self.current_stage}/{self.MAX_STAGES}", True, self.COLOR_UI_TEXT)
        timer_text = self.font_small.render(f"TIME: {self.stage_timer // self.FPS}", True, self.COLOR_UI_TEXT)
        self.screen.blit(stage_text, (self.WIDTH / 2 - stage_text.get_width()/2, 10))
        self.screen.blit(timer_text, (self.WIDTH / 2 - timer_text.get_width()/2, 30))

        # Lives
        for i in range(self.player_lives):
            heart_pos = (self.WIDTH - 30 - i * 25, 15)
            pygame.draw.circle(self.screen, self.COLOR_PLAYER, (heart_pos[0] - 5, heart_pos[1]), 7)
            pygame.draw.circle(self.screen, self.COLOR_PLAYER, (heart_pos[0] + 5, heart_pos[1]), 7)
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, [(heart_pos[0]-10, heart_pos[1]+2), (heart_pos[0]+10, heart_pos[1]+2), (heart_pos[0], heart_pos[1]+12)])

        # Shield Cooldown
        bar_width = 100
        bar_height = 10
        bar_x = 10
        bar_y = self.HEIGHT - bar_height - 10
        
        cooldown_ratio = self.player_shield_cooldown / self.PLAYER_SHIELD_COOLDOWN_MAX
        fill_width = bar_width * (1 - cooldown_ratio)
        
        pygame.draw.rect(self.screen, self.COLOR_UI_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        if fill_width > 0:
            pygame.draw.rect(self.screen, self.COLOR_UI_BAR_FG, (bar_x, bar_y, fill_width, bar_height))
        shield_text = self.font_small.render("SHIELD", True, self.COLOR_UI_TEXT)
        self.screen.blit(shield_text, (bar_x + bar_width + 5, bar_y))

        # Game Over / Win Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.player_lives <= 0:
                msg = "GAME OVER"
            else:
                msg = "YOU WIN!"
            
            end_text = self.font_large.render(msg, True, self.COLOR_UI_TEXT)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "stage": self.current_stage,
            "lives": self.player_lives,
            "aliens_remaining": len(self.aliens)
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Space Invaders")
    clock = pygame.time.Clock()
    
    terminated = False
    total_reward = 0
    
    print("\n" + "="*30)
    print(env.game_description)
    print(env.user_guide)
    print("="*30 + "\n")

    while not terminated:
        movement = 0 # no-op
        space_held = 0
        shift_held = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(env.FPS)

    print(f"Game Over! Final Score: {info['score']:.2f}, Total Steps: {info['steps']}")
    env.close()