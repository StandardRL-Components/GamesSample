import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import os
import os
import pygame


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    """
    A top-down arcade shooter where the player must destroy waves of descending aliens.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    user_guide = (
        "Controls: Use arrow keys to move. Press Space to fire your weapon. "
        "Survive all waves to win."
    )

    game_description = (
        "A fast-paced, top-down arcade shooter. Destroy waves of descending aliens "
        "while dodging their projectiles. Score points for each kill and survive to win."
    )

    auto_advance = True

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.screen_width = 640
        self.screen_height = 400

        self.observation_space = Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        # --- Colors ---
        self.COLOR_BG = (10, 10, 20)
        self.COLOR_PLAYER = (50, 255, 150)
        self.COLOR_PLAYER_PROJ = (200, 255, 220)
        self.COLOR_ALIEN_PROJ = (255, 100, 100)
        self.COLOR_UI_TEXT = (230, 230, 230)
        self.COLOR_EXPLOSION = (255, 165, 0)
        self.WAVE_COLORS = [
            (255, 80, 80),   # Wave 1: Red
            (80, 150, 255),  # Wave 2: Blue
            (255, 255, 80),  # Wave 3: Yellow
            (200, 80, 255),  # Wave 4: Purple
            (80, 255, 200),  # Wave 5: Cyan
        ]

        # --- Fonts ---
        self.font_ui = pygame.font.SysFont("consolas", 20, bold=True)
        self.font_game_over = pygame.font.SysFont("impact", 64)

        # --- Game Parameters ---
        self.PLAYER_SPEED = 7
        self.PLAYER_HEALTH_MAX = 100
        self.PLAYER_SIZE = 12
        self.PROJECTILE_SPEED = 12
        self.MAX_STEPS = 1500  # 50 seconds at 30fps
        self.TOTAL_WAVES = 5

        # Initialize state variables to default values. They will be properly reset in reset().
        self.player_pos = [self.screen_width / 2, self.screen_height - 50]
        self.player_health = self.PLAYER_HEALTH_MAX
        self.player_projectiles = []
        self.aliens = []
        self.alien_projectiles = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False
        self.wave_number = 1
        self.last_space_held = False
        
        # This is needed to correctly initialize the np_random generator for the first time
        super().reset(seed=None)

        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win_condition = False

        self.player_pos = [self.screen_width / 2, self.screen_height - 50]
        self.player_health = self.PLAYER_HEALTH_MAX

        self.player_projectiles = []
        self.aliens = []
        self.alien_projectiles = []
        self.particles = []

        self.wave_number = 1
        self.last_space_held = False

        self._spawn_wave()

        return self._get_observation(), self._get_info()

    def step(self, action):
        reward = -0.01  # Small penalty for each step to encourage efficiency

        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1

        # --- Update Game Logic ---
        self._handle_player_input(movement, space_held)
        self._update_aliens()
        reward += self._update_projectiles()
        self._update_particles()

        # --- Check Game State ---
        if not self.aliens and self.wave_number <= self.TOTAL_WAVES and not self.win_condition:
            reward += 10  # Wave clear bonus
            self.wave_number += 1
            if self.wave_number > self.TOTAL_WAVES:
                self.win_condition = True
                self.game_over = True
                reward += 100  # Win bonus
            else:
                self._spawn_wave()

        # --- Check Termination Conditions ---
        terminated = False
        if self.player_health <= 0:
            if not self.game_over:
                reward -= 100  # Game over penalty
                self._create_explosion(self.player_pos, 100, self.COLOR_EXPLOSION)
            self.game_over = True
        
        if self.steps >= self.MAX_STEPS:
            self.game_over = True

        if self.game_over:
            terminated = True

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _spawn_wave(self):
        num_aliens_x = 7 + self.wave_number
        num_aliens_y = 1 + (self.wave_number // 2)
        alien_size = 20
        spacing = 45
        grid_width = (num_aliens_x - 1) * spacing
        start_x = (self.screen_width - grid_width) / 2

        for row in range(num_aliens_y):
            for col in range(num_aliens_x):
                x = start_x + col * spacing
                y = 50 + row * spacing
                self.aliens.append({
                    'rect': pygame.Rect(x, y, alien_size, alien_size),
                    'color': self.WAVE_COLORS[self.wave_number - 1],
                    'original_x': x,
                    'phase': self.np_random.random() * 2 * math.pi
                })

    def _handle_player_input(self, movement, space_held):
        # Movement
        if movement == 1: self.player_pos[1] -= self.PLAYER_SPEED  # Up
        if movement == 2: self.player_pos[1] += self.PLAYER_SPEED  # Down
        if movement == 3: self.player_pos[0] -= self.PLAYER_SPEED  # Left
        if movement == 4: self.player_pos[0] += self.PLAYER_SPEED  # Right

        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.screen_width)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.screen_height)

        # Firing
        if space_held and not self.last_space_held:
            # Sound: Player shoot
            proj_pos = [self.player_pos[0], self.player_pos[1] - self.PLAYER_SIZE]
            self.player_projectiles.append(pygame.Rect(proj_pos[0] - 2, proj_pos[1] - 10, 5, 12))
        self.last_space_held = space_held

    def _update_aliens(self):
        fire_freq = 0.002 + (self.wave_number - 1) * 0.001
        for alien in self.aliens[:]:
            alien['rect'].x = alien['original_x'] + math.sin(self.steps * 0.03 + alien['phase']) * 50
            alien['rect'].y += 0.4  # Gradual descent

            if self.np_random.random() < fire_freq:
                # Sound: Alien shoot
                proj_pos = alien['rect'].midbottom
                self.alien_projectiles.append(pygame.Rect(proj_pos[0] - 2, proj_pos[1], 5, 12))

            if alien['rect'].top > self.screen_height:
                self.aliens.remove(alien)
                self.player_health -= 20 # Penalty for letting one pass

    def _update_projectiles(self):
        reward = 0
        player_rect = pygame.Rect(self.player_pos[0] - self.PLAYER_SIZE, self.player_pos[1] - self.PLAYER_SIZE, self.PLAYER_SIZE*2, self.PLAYER_SIZE*2)

        # Player projectiles
        for proj in self.player_projectiles[:]:
            proj.y -= self.PROJECTILE_SPEED
            if proj.bottom < 0:
                self.player_projectiles.remove(proj)
                continue
            
            hit_alien_indices = proj.collidelistall([a['rect'] for a in self.aliens])
            if hit_alien_indices:
                self.player_projectiles.remove(proj)
                for i in sorted(hit_alien_indices, reverse=True):
                    # Sound: Alien explosion
                    self._create_explosion(self.aliens[i]['rect'].center, 30, self.COLOR_EXPLOSION)
                    del self.aliens[i]
                    self.score += 10
                    reward += 1
                break

        # Alien projectiles
        for proj in self.alien_projectiles[:]:
            proj.y += self.PROJECTILE_SPEED * 0.7
            if proj.top > self.screen_height:
                self.alien_projectiles.remove(proj)
                continue
            
            if not self.game_over and player_rect.colliderect(proj):
                self.alien_projectiles.remove(proj)
                self.player_health -= 10
                reward -= 1
                # Sound: Player hit
                self._create_explosion(self.player_pos, 20, self.COLOR_PLAYER)

        return reward

    def _create_explosion(self, pos, count, color):
        for _ in range(count):
            angle = self.np_random.random() * 2 * math.pi
            speed = self.np_random.random() * 4 + 1
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 35)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 'max_life': lifespan, 'color': color})

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] <= 0:
                self.particles.remove(p)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Particles
        for p in self.particles:
            life_ratio = p['lifespan'] / p['max_life']
            alpha = int(255 * life_ratio)
            radius = int(2 * life_ratio)
            if radius > 0:
                # Create a temporary surface for the particle to handle alpha blending
                temp_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                pygame.gfxdraw.filled_circle(temp_surf, radius, radius, radius, (*p['color'], alpha))
                self.screen.blit(temp_surf, (int(p['pos'][0]) - radius, int(p['pos'][1]) - radius))


        # Projectiles
        for proj in self.player_projectiles: pygame.draw.rect(self.screen, self.COLOR_PLAYER_PROJ, proj)
        for proj in self.alien_projectiles: pygame.draw.rect(self.screen, self.COLOR_ALIEN_PROJ, proj)

        # Aliens
        for alien in self.aliens: pygame.draw.rect(self.screen, alien['color'], alien['rect'])

        # Player
        if self.player_health > 0:
            p = self.player_pos
            s = self.PLAYER_SIZE
            points = [(p[0], p[1] - s), (p[0] - s, p[1] + s), (p[0] + s, p[1] + s)]
            pygame.gfxdraw.aapolygon(self.screen, [(int(x), int(y)) for x, y in points], self.COLOR_PLAYER)
            pygame.gfxdraw.filled_polygon(self.screen, [(int(x), int(y)) for x, y in points], self.COLOR_PLAYER)

    def _render_ui(self):
        # Health bar
        health_ratio = np.clip(self.player_health / self.PLAYER_HEALTH_MAX, 0, 1)
        bar_width = 200
        bar_height = 20
        pygame.draw.rect(self.screen, (50, 50, 50), (10, 10, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, (10, 10, int(bar_width * health_ratio), bar_height))

        # Score
        score_text = self.font_ui.render(f"SCORE: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.screen_width - score_text.get_width() - 10, 10))

        # Wave
        wave_str = f"WAVE: {self.wave_number}/{self.TOTAL_WAVES}" if self.wave_number <= self.TOTAL_WAVES else "CLEAR!"
        wave_text = self.font_ui.render(wave_str, True, self.COLOR_UI_TEXT)
        self.screen.blit(wave_text, (self.screen_width / 2 - wave_text.get_width() / 2, 10))

        # Game Over / Win message
        if self.game_over:
            msg = "YOU WIN!" if self.win_condition else "GAME OVER"
            color = (50, 255, 50) if self.win_condition else (255, 50, 50)
            end_text = self.font_game_over.render(msg, True, color)
            pos = (self.screen_width / 2 - end_text.get_width() / 2, self.screen_height / 2 - end_text.get_height() / 2)
            self.screen.blit(end_text, pos)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "health": self.player_health,
            "wave": self.wave_number,
            "aliens_remaining": len(self.aliens)
        }

    def close(self):
        pygame.font.quit()
        pygame.quit()

    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.screen_height, self.screen_width, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.screen_height, self.screen_width, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        

if __name__ == '__main__':
    # This block allows you to play the game directly
    # To use, you might need to unset the dummy video driver
    # and install pygame dependencies:
    # unset SDL_VIDEODRIVER
    # pip install pygame
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Arcade Shooter")
    clock = pygame.time.Clock()

    running = True
    while running:
        movement = 0 # No-op
        space = 0
        shift = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
        
        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Convert numpy array to pygame surface for display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']}")
            pygame.time.wait(3000) # Pause for 3 seconds
            obs, info = env.reset()

        clock.tick(env.metadata["render_fps"])

    env.close()