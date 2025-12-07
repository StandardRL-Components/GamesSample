
# Generated: 2025-08-28T02:43:42.978421
# Source Brief: brief_04540.md
# Brief Index: 4540

        
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
        "Controls: ←→ to move. Press space to fire your weapon."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Fend off waves of descending aliens in a retro-inspired, procedurally generated top-down shooter."
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
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 64)
        
        # Colors
        self.COLOR_BG = (0, 0, 10)
        self.COLOR_PLAYER = (50, 255, 50)
        self.COLOR_PROJECTILE = (255, 255, 255)
        self.COLOR_EXPLOSION = (255, 255, 255)
        self.COLOR_UI = (220, 220, 220)
        self.WAVE_COLORS = [
            (0, 150, 255),  # Wave 1: Blue
            (0, 255, 255),  # Wave 2: Cyan
            (255, 255, 0),  # Wave 3: Yellow
            (255, 150, 0),  # Wave 4: Orange
            (255, 50, 50),   # Wave 5: Red
        ]

        # Game constants
        self.MAX_STEPS = 10000
        self.TOTAL_WAVES = 5
        self.PLAYER_SPEED = 8
        self.PROJECTILE_SPEED = 12
        self.FIRE_COOLDOWN_FRAMES = 6
        self.ALIEN_SIZE = 20
        self.PLAYER_SIZE = 15
        
        # Initialize state variables
        self.reset()

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = [self.WIDTH / 2, self.HEIGHT - 40]
        self.aliens = []
        self.projectiles = []
        self.explosions = []
        
        self.steps = 0
        self.score = 0
        self.current_wave = 1
        self.game_over = False
        self.win = False
        self.fire_cooldown = 0
        
        self._spawn_wave()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = 0.0

        if self.game_over:
            # If the game is over, no actions should change the state
            terminated = True
        else:
            self.steps += 1
            reward += 0.001 # Small reward for surviving each frame

            # Unpack factorized action
            movement = action[0]  # 0-4: none/up/down/left/right
            space_held = action[1] == 1  # Boolean
            
            self._handle_input(movement, space_held)
            self._update_projectiles()
            self._update_aliens()
            self._update_explosions()
            
            collision_reward = self._handle_collisions()
            reward += collision_reward

            self._check_wave_completion()
            
            terminated = self.game_over or self.steps >= self.MAX_STEPS
            if terminated:
                if self.win:
                    reward += 100
                elif self.game_over: # Loss condition
                    reward -= 100
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _handle_input(self, movement, space_held):
        # Player movement
        if movement == 3:  # Left
            self.player_pos[0] -= self.PLAYER_SPEED
        elif movement == 4:  # Right
            self.player_pos[0] += self.PLAYER_SPEED
        self.player_pos[0] = np.clip(self.player_pos[0], self.PLAYER_SIZE, self.WIDTH - self.PLAYER_SIZE)

        # Firing
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1
            
        if space_held and self.fire_cooldown == 0:
            # Fire projectile from the tip of the ship
            self.projectiles.append(pygame.Rect(self.player_pos[0] - 2, self.player_pos[1] - self.PLAYER_SIZE, 4, 10))
            self.fire_cooldown = self.FIRE_COOLDOWN_FRAMES
            # Sound effect placeholder: # Player shoot sound (pew!)

    def _update_projectiles(self):
        for p in self.projectiles[:]:
            p.y -= self.PROJECTILE_SPEED
            if p.bottom < 0:
                self.projectiles.remove(p)

    def _update_aliens(self):
        alien_base_speed = 0.5 + (self.current_wave - 1) * 0.4
        
        for alien in self.aliens:
            time_alive = self.steps - alien['spawn_step']
            
            if alien['pattern'] == 'straight':
                alien['pos'][1] += alien_base_speed
            elif alien['pattern'] == 'zigzag':
                alien['pos'][1] += alien_base_speed
                alien['pos'][0] = alien['start_x'] + math.sin(time_alive * 0.05) * 50
            elif alien['pattern'] == 'bounce':
                alien['pos'][0] += alien['dx'] * (2 + self.current_wave * 0.5)
                alien['pos'][1] += alien_base_speed * 0.8
                if alien['pos'][0] <= self.ALIEN_SIZE/2 or alien['pos'][0] >= self.WIDTH - self.ALIEN_SIZE/2:
                    alien['dx'] *= -1

            alien['rect'].center = alien['pos']

            if alien['rect'].bottom >= self.HEIGHT:
                self.game_over = True
                # Sound effect placeholder: # Game over sound (klaxon)

    def _update_explosions(self):
        for exp in self.explosions[:]:
            exp['radius'] += exp['speed']
            if exp['radius'] > exp['max_radius']:
                self.explosions.remove(exp)

    def _handle_collisions(self):
        collision_reward = 0
        for p in self.projectiles[:]:
            for a in self.aliens[:]:
                if p.colliderect(a['rect']):
                    # Sound effect placeholder: # Alien hit/explosion sound
                    self.explosions.append({
                        'pos': a['rect'].center,
                        'radius': 5,
                        'max_radius': 25,
                        'speed': 2
                    })
                    self.aliens.remove(a)
                    if p in self.projectiles: self.projectiles.remove(p)
                    self.score += 10
                    collision_reward += 1
                    break
        return collision_reward

    def _check_wave_completion(self):
        if not self.aliens and not self.game_over:
            if self.current_wave < self.TOTAL_WAVES:
                self.current_wave += 1
                self._spawn_wave()
            else:
                self.win = True
                self.game_over = True
                # Sound effect placeholder: # Victory fanfare

    def _spawn_wave(self):
        num_aliens = 8 + self.current_wave * 2
        rows = 2 + self.current_wave // 2
        cols = (num_aliens + rows -1) // rows

        patterns = ['straight', 'zigzag', 'straight', 'zigzag', 'bounce']
        pattern = patterns[self.current_wave - 1]

        for i in range(num_aliens):
            row = i // cols
            col = i % cols
            
            start_x = (self.WIDTH / (cols + 1)) * (col + 1)
            start_y = -30 - row * (self.ALIEN_SIZE + 20)
            
            alien = {
                'pos': [start_x, start_y],
                'rect': pygame.Rect(0, 0, self.ALIEN_SIZE, self.ALIEN_SIZE),
                'pattern': pattern,
                'spawn_step': self.steps,
                'start_x': start_x,
                'dx': random.choice([-1, 1]) # for bounce pattern
            }
            alien['rect'].center = alien['pos']
            self.aliens.append(alien)
            
    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw aliens
        alien_color = self.WAVE_COLORS[self.current_wave - 1]
        for alien in self.aliens:
            pygame.draw.rect(self.screen, alien_color, alien['rect'])

        # Draw projectiles
        for p in self.projectiles:
            pygame.draw.rect(self.screen, self.COLOR_PROJECTILE, p)

        # Draw player
        p1 = (self.player_pos[0], self.player_pos[1] - self.PLAYER_SIZE)
        p2 = (self.player_pos[0] - self.PLAYER_SIZE, self.player_pos[1] + self.PLAYER_SIZE / 2)
        p3 = (self.player_pos[0] + self.PLAYER_SIZE, self.player_pos[1] + self.PLAYER_SIZE / 2)
        points = [p1, p2, p3]
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_PLAYER)

        # Draw explosions
        for exp in self.explosions:
            alpha = 255 * (1 - exp['radius'] / exp['max_radius'])
            color = (*self.COLOR_EXPLOSION, int(alpha))
            pygame.gfxdraw.aacircle(self.screen, int(exp['pos'][0]), int(exp['pos'][1]), int(exp['radius']), color)

    def _render_ui(self):
        # Score
        score_surf = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI)
        self.screen.blit(score_surf, (10, 10))

        # Wave
        wave_surf = self.font_ui.render(f"WAVE: {self.current_wave} / {self.TOTAL_WAVES}", True, self.COLOR_UI)
        self.screen.blit(wave_surf, (self.WIDTH - wave_surf.get_width() - 10, 10))

        # Game Over / Win message
        if self.game_over:
            if self.win:
                msg = "YOU WIN!"
                color = (0, 255, 128)
            else:
                msg = "GAME OVER"
                color = (255, 0, 0)
            
            text_surf = self.font_game_over.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(text_surf, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "wave": self.current_wave,
            "win": self.win,
        }

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

    def close(self):
        pygame.quit()

# Example usage for visualization and testing
if __name__ == '__main__':
    import os
    # Set a non-dummy video driver if you want to see the window
    if os.environ.get("SDL_VIDEODRIVER", "") != "dummy":
        os.environ['SDL_VIDEODRIVER'] = 'x11' # Or 'windows', 'cocoa'

    env = GameEnv(render_mode="rgb_array")
    
    # Use Pygame for human interaction
    pygame.display.set_caption("Retro Alien Shooter")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    
    obs, info = env.reset()
    terminated = False
    
    # Game loop for human play
    while not terminated:
        # Action defaults
        movement = 0 # No-op
        space_held = 0 # Released
        shift_held = 0 # Released

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            movement = 3
        elif keys[pygame.K_RIGHT]:
            movement = 4
        
        if keys[pygame.K_SPACE]:
            space_held = 1
        
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            shift_held = 1
            
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Control frame rate

    env.close()