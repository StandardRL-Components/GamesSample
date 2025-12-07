
# Generated: 2025-08-27T16:34:10.146672
# Source Brief: brief_01264.md
# Brief Index: 1264

        
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

    # Must be a short, user-facing control string:
    user_guide = "Controls: Use arrow keys to move. Press space to shoot."

    # Must be a short, user-facing description of the game:
    game_description = "A retro-style top-down shooter. Destroy all 5 waves of aliens to win."

    # Should frames auto-advance or wait for user input?
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 10000
    TOTAL_WAVES = 5

    # Colors
    COLOR_BG = (15, 15, 25)
    COLOR_PLAYER = (50, 255, 50)
    COLOR_PLAYER_GLOW = (50, 255, 50, 50)
    COLOR_ALIEN = (255, 50, 50)
    COLOR_ALIEN_GLOW = (255, 50, 50, 50)
    COLOR_PLAYER_BULLET = (200, 255, 255)
    COLOR_ALIEN_BULLET = (255, 100, 255)
    COLOR_EXPLOSION = (255, 180, 50)
    COLOR_TEXT = (220, 220, 220)
    COLOR_STAR = (200, 200, 220)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 40, bold=True)
        self.font_win_lose = pygame.font.SysFont("monospace", 60, bold=True)

        # Game state variables are initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.np_random = None

        self.player_pos = [0,0]
        self.player_lives = 0
        self.player_speed = 5
        self.player_projectiles = []
        self.player_shoot_cooldown = 0
        self.player_shoot_cooldown_max = 10 # 3 shots per second at 30fps

        self.aliens = []
        self.alien_projectiles = []
        self.alien_projectile_speed = 0

        self.explosions = []
        self.stars = []

        self.current_wave = 1
        
        self._generate_stars()
        self.reset()
        
        # self.validate_implementation() # Uncomment for self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        else:
            self.np_random = np.random.default_rng()


        # Initialize all game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False

        self.player_pos = [self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT - 50]
        self.player_lives = 3
        
        self.player_projectiles = []
        self.player_shoot_cooldown = 0

        self.aliens = []
        self.alien_projectiles = []
        self.explosions = []

        self.current_wave = 1
        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty for each step to encourage efficiency

        if not self.game_over:
            # Unpack factorized action
            movement = action[0]  # 0-4: none/up/down/left/right
            space_held = action[1] == 1  # Boolean
            # shift_held is not used in this game

            # Update game logic
            reward += self._update_player(movement, space_held)
            self._update_projectiles()
            alien_reached_bottom = self._update_aliens()
            self._update_explosions()
            reward += self._handle_collisions()

            # Check for wave clear
            if not self.aliens and not self.game_over:
                reward += 100 # Wave clear bonus
                self.current_wave += 1
                if self.current_wave > self.TOTAL_WAVES:
                    self.win = True
                    self.game_over = True
                    reward += 500 # Game win bonus
                else:
                    self._spawn_wave()
            
            # Check for game over conditions
            if alien_reached_bottom:
                self.player_lives = 0 # Instant loss if aliens get past
                reward -= 50
            
            if self.player_lives <= 0 and not self.game_over:
                self.game_over = True
                self.win = False
                self._create_explosion(self.player_pos, 30)
                # Sound: Player_Explosion.wav

        self.steps += 1
        terminated = self.game_over or self.steps >= self.MAX_STEPS

        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _spawn_wave(self):
        self.aliens.clear()
        self.alien_projectiles.clear()
        
        # Difficulty scaling
        rows = 2 + math.ceil(self.current_wave / 2)
        cols = 8
        self.alien_projectile_speed = 3.0 + (self.current_wave * 0.4)
        alien_fire_rate = max(5, 60 - self.current_wave * 8) # Cooldown ticks
        alien_descent_speed = 0.2 + self.current_wave * 0.05

        for row in range(rows):
            for col in range(cols):
                base_x = 80 + col * 60
                pos_y = 40 + row * 40
                self.aliens.append({
                    "pos": [base_x, pos_y],
                    "base_x": base_x,
                    "alive": True,
                    "shoot_cooldown": self.np_random.integers(0, alien_fire_rate),
                    "fire_rate": alien_fire_rate,
                    "descent_speed": alien_descent_speed,
                    "patrol_offset": self.np_random.random() * 2 * math.pi # Random start in sine wave
                })

    def _update_player(self, movement, space_held):
        # Movement
        if movement == 1: self.player_pos[1] -= self.player_speed
        if movement == 2: self.player_pos[1] += self.player_speed
        if movement == 3: self.player_pos[0] -= self.player_speed
        if movement == 4: self.player_pos[0] += self.player_speed
        
        # Clamp position to screen
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.SCREEN_WIDTH)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.SCREEN_HEIGHT)

        # Shooting
        if self.player_shoot_cooldown > 0:
            self.player_shoot_cooldown -= 1

        if space_held and self.player_shoot_cooldown == 0:
            # Sound: Player_Shoot.wav
            self.player_projectiles.append(list(self.player_pos))
            self.player_shoot_cooldown = self.player_shoot_cooldown_max
        return 0

    def _update_projectiles(self):
        # Player projectiles
        for proj in self.player_projectiles[:]:
            proj[1] -= 8 # Move up
            if proj[1] < 0:
                self.player_projectiles.remove(proj)
        
        # Alien projectiles
        for proj in self.alien_projectiles[:]:
            proj[1] += self.alien_projectile_speed # Move down
            if proj[1] > self.SCREEN_HEIGHT:
                self.alien_projectiles.remove(proj)

    def _update_aliens(self):
        alien_reached_bottom = False
        for alien in self.aliens:
            # Patrol movement (sine wave)
            patrol_amplitude = 20
            patrol_speed = 0.05
            alien['patrol_offset'] += patrol_speed
            alien['pos'][0] = alien['base_x'] + math.sin(alien['patrol_offset']) * patrol_amplitude
            
            # Descent
            alien['pos'][1] += alien['descent_speed']
            if alien['pos'][1] > self.SCREEN_HEIGHT - 20:
                alien_reached_bottom = True

            # Shooting
            if alien['shoot_cooldown'] > 0:
                alien['shoot_cooldown'] -= 1
            else:
                # 1 in 200 chance to shoot each frame if cooldown is ready
                if self.np_random.random() < 0.005:
                    # Sound: Alien_Shoot.wav
                    self.alien_projectiles.append(list(alien['pos']))
                    alien['shoot_cooldown'] = alien['fire_rate']
        return alien_reached_bottom

    def _update_explosions(self):
        for exp in self.explosions[:]:
            exp['radius'] += exp['growth']
            exp['alpha'] -= exp['fade']
            if exp['alpha'] <= 0:
                self.explosions.remove(exp)

    def _handle_collisions(self):
        reward = 0
        
        # Player projectiles vs aliens
        for proj in self.player_projectiles[:]:
            proj_rect = pygame.Rect(proj[0] - 2, proj[1] - 8, 4, 16)
            for alien in self.aliens[:]:
                alien_rect = pygame.Rect(alien['pos'][0] - 12, alien['pos'][1] - 8, 24, 16)
                if proj_rect.colliderect(alien_rect):
                    # Sound: Alien_Hit.wav
                    self.aliens.remove(alien)
                    if proj in self.player_projectiles: self.player_projectiles.remove(proj)
                    self._create_explosion(alien['pos'], 15)
                    self.score += 100
                    reward += 10
                    break

        # Alien projectiles vs player
        player_rect = pygame.Rect(self.player_pos[0] - 10, self.player_pos[1] - 8, 20, 16)
        for proj in self.alien_projectiles[:]:
            proj_rect = pygame.Rect(proj[0] - 2, proj[1] - 8, 4, 16)
            if player_rect.colliderect(proj_rect):
                # Sound: Player_Hit.wav
                self.alien_projectiles.remove(proj)
                self.player_lives -= 1
                reward -= 50
                self._create_explosion(self.player_pos, 20)
                if self.player_lives <= 0:
                    self.game_over = True
                    self.win = False
                break
        return reward

    def _create_explosion(self, pos, max_radius):
        self.explosions.append({
            'pos': list(pos),
            'radius': 2,
            'max_radius': max_radius,
            'growth': 1,
            'alpha': 255,
            'fade': 255 / max_radius
        })

    def _generate_stars(self):
        self.stars = []
        for _ in range(100):
            self.stars.append((
                random.randint(0, self.SCREEN_WIDTH),
                random.randint(0, self.SCREEN_HEIGHT),
                random.choice([1, 1, 1, 2])
            ))

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
        for x, y, size in self.stars:
            pygame.draw.circle(self.screen, self.COLOR_STAR, (x, y), size)
        
        # Aliens
        for alien in self.aliens:
            x, y = int(alien['pos'][0]), int(alien['pos'][1])
            points = [(x - 12, y), (x, y - 8), (x + 12, y), (x + 8, y + 8), (x - 8, y + 8)]
            pygame.draw.polygon(self.screen, self.COLOR_ALIEN, points)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_ALIEN)

        # Player
        if self.player_lives > 0:
            x, y = int(self.player_pos[0]), int(self.player_pos[1])
            points = [(x, y - 10), (x - 8, y + 8), (x + 8, y + 8)]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)

        # Projectiles
        for proj in self.player_projectiles:
            start = (int(proj[0]), int(proj[1]))
            end = (int(proj[0]), int(proj[1] - 10))
            pygame.draw.line(self.screen, self.COLOR_PLAYER_BULLET, start, end, 3)
        
        for proj in self.alien_projectiles:
            start = (int(proj[0]), int(proj[1]))
            end = (int(proj[0]), int(proj[1] + 10))
            pygame.draw.line(self.screen, self.COLOR_ALIEN_BULLET, start, end, 3)

        # Explosions
        for exp in self.explosions:
            pos = (int(exp['pos'][0]), int(exp['pos'][1]))
            radius = int(exp['radius'])
            alpha = int(exp['alpha'])
            if radius > 0 and alpha > 0:
                temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                color = (*self.COLOR_EXPLOSION, alpha)
                pygame.draw.circle(temp_surf, color, (radius, radius), radius)
                self.screen.blit(temp_surf, (pos[0] - radius, pos[1] - radius), special_flags=pygame.BLEND_RGBA_ADD)

        # Win/Loss Message
        if self.game_over:
            msg_text = "YOU WIN!" if self.win else "GAME OVER"
            msg_color = self.COLOR_PLAYER if self.win else self.COLOR_ALIEN
            msg_surf = self.font_win_lose.render(msg_text, True, msg_color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(msg_surf, msg_rect)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Wave
        wave_text = self.font_ui.render(f"WAVE: {self.current_wave}/{self.TOTAL_WAVES}", True, self.COLOR_TEXT)
        wave_rect = wave_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(wave_text, wave_rect)

        # Lives
        lives_text = self.font_ui.render("LIVES:", True, self.COLOR_TEXT)
        self.screen.blit(lives_text, (10, self.SCREEN_HEIGHT - 30))
        for i in range(self.player_lives):
            x, y = 80 + i * 25, self.SCREEN_HEIGHT - 22
            points = [(x, y - 7), (x - 5, y + 5), (x + 5, y + 5)]
            pygame.draw.polygon(self.screen, self.COLOR_PLAYER, points)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "lives": self.player_lives,
            "win": self.win
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

# Example usage:
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    
    # --- To run the game with manual controls ---
    # This loop is for human play and demonstration, not for RL training
    
    # Pygame setup for display
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    obs, info = env.reset()
    done = False
    
    print(env.user_guide)

    while not done:
        # Action mapping from keyboard
        keys = pygame.key.get_pressed()
        movement = 0 # no-op
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        space_held = 1 if keys[pygame.K_SPACE] else 0
        shift_held = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Handle window closing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        clock.tick(30) # Limit to 30 FPS

    print(f"Game Over! Final Score: {info['score']}, Win: {info['win']}")
    env.close()