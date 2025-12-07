
# Generated: 2025-08-27T15:37:24.220932
# Source Brief: brief_01030.md
# Brief Index: 1030

        
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
        "Controls: Arrow keys to move your ship. Hold space to fire. Press shift to use a bomb."
    )

    # Must be a short,user-facing description of the game:
    game_description = (
        "Defend Earth from waves of descending aliens in this retro top-down arcade shooter."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    # Game settings
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    FPS = 30
    MAX_STEPS = 5000
    TOTAL_WAVES = 3
    TOTAL_ALIENS = 50

    # Colors
    COLOR_BG = (10, 5, 20)
    COLOR_PLAYER = (0, 255, 128)
    COLOR_PLAYER_PROJECTILE = (150, 255, 255)
    COLOR_ALIEN = (255, 50, 50)
    COLOR_ALIEN_PROJECTILE = (255, 100, 255)
    COLOR_EXPLOSION_1 = (255, 200, 0)
    COLOR_EXPLOSION_2 = (255, 100, 0)
    COLOR_POWERUP = (50, 150, 255)
    COLOR_TEXT = (220, 220, 220)
    COLOR_UI_ACCENT = (0, 255, 128)

    # Player settings
    PLAYER_SPEED = 8
    PLAYER_FIRE_COOLDOWN = 6  # frames
    PLAYER_LIVES = 3

    # Powerup settings
    POWERUP_DROP_CHANCE = 0.15
    POWERUP_SPEED = 2
    BOMB_POWERUP_ID = 'bomb'

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
        self.font_large = pygame.font.Font(None, 72)
        
        self.np_random = None
        self.starfield = None
        
        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.np_random is None:
             self.np_random = np.random.default_rng(seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        
        self.player_pos = [self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 50]
        self.player_lives = self.PLAYER_LIVES
        self.player_projectiles = []
        self.player_fire_cooldown_timer = 0
        self.player_invincibility_timer = 0
        
        self.bombs = 1 # Start with one bomb

        self.aliens = []
        self.enemy_projectiles = []
        self.alien_fire_rate = 1.0 # seconds
        self.alien_descent_speed = 0.2 # pixels/frame
        self.alien_fire_cooldown_timer = 0

        self.explosions = []
        self.powerups = []

        self.current_wave = 1
        self.wave_transition_timer = 0

        self.last_space_held = False
        self.last_shift_held = False

        self._create_starfield()
        self._setup_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = -0.01  # Small penalty for time passing

        if not self.game_over and not self.game_won:
            if self.wave_transition_timer > 0:
                self.wave_transition_timer -= 1
                if self.wave_transition_timer == 0:
                    self._setup_wave()
            else:
                reward += self._update_game_logic(movement, space_held, shift_held)

        self.steps += 1
        terminated = self.game_over or self.game_won or self.steps >= self.MAX_STEPS
        
        if terminated and not self.game_over and not self.game_won: # Max steps reached
            self.game_over = True

        if self.game_over:
            reward -= 100
        elif self.game_won:
            reward += 500

        self.last_space_held = space_held
        self.last_shift_held = shift_held

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _update_game_logic(self, movement, space_held, shift_held):
        reward = 0
        
        # Cooldowns
        if self.player_fire_cooldown_timer > 0: self.player_fire_cooldown_timer -= 1
        if self.player_invincibility_timer > 0: self.player_invincibility_timer -= 1
        if self.alien_fire_cooldown_timer > 0: self.alien_fire_cooldown_timer -= 1

        # Player actions
        self._handle_player_movement(movement)
        self._handle_player_firing(space_held)
        if self._handle_bomb_usage(shift_held):
            reward += len(self.aliens) * 0.5 # Reward for bomb usage
            self.aliens.clear()
            self.enemy_projectiles.clear()

        # Update entities
        self._update_projectiles()
        self._update_aliens()
        self._update_powerups()
        self._update_explosions()

        # Collisions
        reward += self._handle_collisions()

        # Check for wave clear
        if not self.aliens and not self.game_won:
            if self.current_wave < self.TOTAL_WAVES:
                self.current_wave += 1
                self.wave_transition_timer = self.FPS * 3 # 3 second transition
                reward += 100
            else:
                self.game_won = True
        
        return reward

    def _handle_player_movement(self, movement):
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED # Up
        if movement == 2: self.player_pos[1] += self.PLAYER_SPEED # Down
        if movement == 3: self.player_pos[0] -= self.PLAYER_SPEED # Left
        if movement == 4: self.player_pos[0] += self.PLAYER_SPEED # Right
        
        self.player_pos[0] = np.clip(self.player_pos[0], 10, self.SCREEN_WIDTH - 10)
        self.player_pos[1] = np.clip(self.player_pos[1], 10, self.SCREEN_HEIGHT - 10)

    def _handle_player_firing(self, space_held):
        if space_held and self.player_fire_cooldown_timer == 0:
            # Sfx: Player shoot
            self.player_projectiles.append(pygame.Rect(self.player_pos[0] - 2, self.player_pos[1] - 20, 4, 15))
            self.player_fire_cooldown_timer = self.PLAYER_FIRE_COOLDOWN

    def _handle_bomb_usage(self, shift_held):
        # Use on key press, not hold
        if shift_held and not self.last_shift_held and self.bombs > 0:
            self.bombs -= 1
            # Sfx: Big explosion
            self._create_explosion(self.player_pos, 200, 60, is_bomb=True)
            return True
        return False
        
    def _update_projectiles(self):
        self.player_projectiles = [p for p in self.player_projectiles if p.y > -20]
        for p in self.player_projectiles: p.y -= 12
        
        self.enemy_projectiles = [p for p in self.enemy_projectiles if p.y < self.SCREEN_HEIGHT + 20]
        for p in self.enemy_projectiles: p.y += 5

    def _update_aliens(self):
        for alien in self.aliens:
            alien['pos'][1] += self.alien_descent_speed
            alien['rect'].topleft = alien['pos']
            if alien['rect'].bottom > self.SCREEN_HEIGHT:
                self.aliens.remove(alien)
                self._player_hit()

        if self.alien_fire_cooldown_timer == 0 and self.aliens:
            firing_alien = self.np_random.choice(self.aliens)
            # Sfx: Enemy shoot
            self.enemy_projectiles.append(pygame.Rect(firing_alien['rect'].centerx - 3, firing_alien['rect'].bottom, 6, 12))
            self.alien_fire_cooldown_timer = int(self.alien_fire_rate * self.FPS)

    def _update_powerups(self):
        for powerup in self.powerups:
            powerup['pos'][1] += self.POWERUP_SPEED
            powerup['rect'].topleft = powerup['pos']
            if powerup['rect'].top > self.SCREEN_HEIGHT:
                self.powerups.remove(powerup)

    def _update_explosions(self):
        for exp in self.explosions:
            exp['life'] -= 1
            exp['radius'] += exp['growth']
        self.explosions = [exp for exp in self.explosions if exp['life'] > 0]

    def _handle_collisions(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos[0] - 12, self.player_pos[1] - 10, 24, 20)

        # Player projectiles vs aliens
        for proj in self.player_projectiles[:]:
            for alien in self.aliens[:]:
                if proj.colliderect(alien['rect']):
                    reward += 0.1 # Hit reward
                    self.player_projectiles.remove(proj)
                    self.aliens.remove(alien)
                    self._create_explosion(alien['rect'].center, 20, 20)
                    self.score += 10
                    reward += 1 # Destroy reward
                    if self.np_random.random() < self.POWERUP_DROP_CHANCE:
                        self._create_powerup(alien['rect'].center)
                    break
        
        # Enemy projectiles vs player
        if self.player_invincibility_timer == 0:
            for proj in self.enemy_projectiles[:]:
                if player_rect.colliderect(proj):
                    self.enemy_projectiles.remove(proj)
                    self._player_hit()
                    break
        
        # Player vs powerups
        for powerup in self.powerups[:]:
            if player_rect.colliderect(powerup['rect']):
                # Sfx: Powerup collect
                self.powerups.remove(powerup)
                if powerup['type'] == self.BOMB_POWERUP_ID:
                    self.bombs = min(self.bombs + 1, 3) # Max 3 bombs
                    self.score += 25
                break
        
        return reward

    def _player_hit(self):
        # Sfx: Player explosion
        self.player_lives -= 1
        self._create_explosion(self.player_pos, 50, 40)
        self.player_invincibility_timer = self.FPS * 2 # 2 seconds invincibility
        if self.player_lives <= 0:
            self.game_over = True
        else: # Reset position
            self.player_pos = [self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT - 50]

    def _setup_wave(self):
        self.aliens.clear()
        self.alien_fire_rate = max(0.2, 1.0 - (self.current_wave - 1) * 0.25)
        self.alien_descent_speed = 0.2 + (self.current_wave - 1) * 0.1
        
        num_aliens_per_row = 10
        num_rows = [2, 3, 3][self.current_wave - 1]
        
        for row in range(num_rows):
            for col in range(num_aliens_per_row):
                x = 60 + col * 50
                y = 50 + row * 40
                self.aliens.append({'pos': [x, y], 'rect': pygame.Rect(x, y, 30, 20)})
    
    def _create_starfield(self):
        self.starfield = []
        for _ in range(200):
            self.starfield.append({
                'pos': [self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT)],
                'speed': self.np_random.random() * 0.5 + 0.1,
                'size': int(self.np_random.integers(1, 3)),
                'color': self.np_random.integers(50, 150)
            })

    def _create_explosion(self, pos, max_radius, life, is_bomb=False):
        growth_rate = max_radius / life
        self.explosions.append({'pos': list(pos), 'radius': 1, 'max_radius': max_radius, 'life': life, 'max_life': life, 'growth': growth_rate, 'is_bomb': is_bomb})

    def _create_powerup(self, pos):
        self.powerups.append({
            'pos': list(pos),
            'rect': pygame.Rect(pos[0]-10, pos[1]-10, 20, 20),
            'type': self.BOMB_POWERUP_ID
        })

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Starfield
        for star in self.starfield:
            star['pos'][1] = (star['pos'][1] + star['speed']) % self.SCREEN_HEIGHT
            c = star['color']
            pygame.draw.circle(self.screen, (c,c,c), (int(star['pos'][0]), int(star['pos'][1])), star['size'])

        # Powerups
        for powerup in self.powerups:
            pygame.draw.rect(self.screen, self.COLOR_POWERUP, powerup['rect'], border_radius=5)
            pygame.draw.rect(self.screen, self.COLOR_TEXT, powerup['rect'], width=2, border_radius=5)
            bomb_text = self.font_small.render("B", True, self.COLOR_TEXT)
            self.screen.blit(bomb_text, bomb_text.get_rect(center=powerup['rect'].center))

        # Aliens
        for alien in self.aliens:
            pygame.draw.ellipse(self.screen, self.COLOR_ALIEN, alien['rect'])

        # Player
        if self.player_lives > 0:
            player_alpha = 255 if (self.player_invincibility_timer // 3) % 2 == 0 else 100
            if player_alpha < 255:
                player_surf = pygame.Surface((24, 24), pygame.SRCALPHA)
                pygame.draw.polygon(player_surf, (*self.COLOR_PLAYER, player_alpha), [(12, 0), (0, 20), (24, 20)])
                self.screen.blit(player_surf, (self.player_pos[0] - 12, self.player_pos[1] - 12))
            else:
                pygame.draw.polygon(self.screen, self.COLOR_PLAYER, [
                    (self.player_pos[0], self.player_pos[1] - 12),
                    (self.player_pos[0] - 12, self.player_pos[1] + 8),
                    (self.player_pos[0] + 12, self.player_pos[1] + 8)
                ])

        # Projectiles
        for p in self.player_projectiles: pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJECTILE, p, border_radius=2)
        for p in self.enemy_projectiles: pygame.draw.rect(self.screen, self.COLOR_ALIEN_PROJECTILE, p, border_radius=2)

        # Explosions
        for exp in self.explosions:
            progress = exp['life'] / exp['max_life']
            current_radius = int(exp['radius'])
            if exp['is_bomb']:
                alpha = int(255 * progress)
                pygame.gfxdraw.filled_circle(self.screen, int(exp['pos'][0]), int(exp['pos'][1]), current_radius, (*(255,255,255), alpha))
            else:
                alpha = int(255 * progress)
                pygame.gfxdraw.filled_circle(self.screen, int(exp['pos'][0]), int(exp['pos'][1]), current_radius, (*self.COLOR_EXPLOSION_1, alpha))
                pygame.gfxdraw.filled_circle(self.screen, int(exp['pos'][0]), int(exp['pos'][1]), int(current_radius*0.5), (*self.COLOR_EXPLOSION_2, alpha))

    def _render_ui(self):
        # Score
        score_text = self.font_small.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Lives
        for i in range(self.player_lives):
            pygame.draw.polygon(self.screen, self.COLOR_UI_ACCENT, [
                (self.SCREEN_WIDTH - 20 - i*25, 10),
                (self.SCREEN_WIDTH - 30 - i*25, 25),
                (self.SCREEN_WIDTH - 10 - i*25, 25)
            ])
            
        # Bombs
        for i in range(self.bombs):
            pygame.draw.circle(self.screen, self.COLOR_POWERUP, (self.SCREEN_WIDTH - 20 - i*25, 45), 8)
            pygame.draw.circle(self.screen, self.COLOR_TEXT, (self.SCREEN_WIDTH - 20 - i*25, 45), 8, 1)

        # Wave
        wave_text = self.font_small.render(f"WAVE: {self.current_wave}/{self.TOTAL_WAVES}", True, self.COLOR_TEXT)
        self.screen.blit(wave_text, wave_text.get_rect(centerx=self.SCREEN_WIDTH // 2, y=10))

        # Game Over / Win / Wave Clear messages
        if self.game_over:
            msg = self.font_large.render("GAME OVER", True, self.COLOR_ALIEN)
            self.screen.blit(msg, msg.get_rect(center=(self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2)))
        elif self.game_won:
            msg = self.font_large.render("YOU WIN!", True, self.COLOR_PLAYER)
            self.screen.blit(msg, msg.get_rect(center=(self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2)))
        elif self.wave_transition_timer > 0:
            alpha = min(255, int(255 * (1.0 - abs(self.wave_transition_timer - self.FPS*1.5) / (self.FPS*1.5))))
            msg_surf = self.font_large.render(f"WAVE {self.current_wave} CLEARED", True, self.COLOR_UI_ACCENT)
            msg_surf.set_alpha(alpha)
            self.screen.blit(msg_surf, msg_surf.get_rect(center=(self.SCREEN_WIDTH//2, self.SCREEN_HEIGHT//2)))

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "lives": self.player_lives,
            "wave": self.current_wave,
            "bombs": self.bombs,
        }

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
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    pygame.display.set_caption("Arcade Defender")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    
    terminated = False
    running = True
    total_reward = 0
    
    while running:
        # --- Human Controls ---
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
        if not terminated:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # --- Pygame Event Loop ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    terminated = False
                    total_reward = 0
                if event.key == pygame.K_ESCAPE:
                    running = False

        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    env.close()
    print(f"Game finished. Final Score: {info['score']}, Total Reward: {total_reward:.2f}")